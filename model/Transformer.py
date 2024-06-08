import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
        batch_first: the batch first option (default=False).
    Shape:
        - Input: x, shape=[sequence length, batch size, embed dim] or [batch size, sequence length, embed dim](batch_first).
        - Output: x, shape=[sequence length, batch size, embed dim] or [batch size, sequence length, embed dim](batch_first).

    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim] or [batch size, sequence length, embed dim](batch_first).
            output: [sequence length, batch size, embed dim] or [batch size, sequence length, embed dim](batch_first).
        Examples:
            >>> output = pos_encoder(x)
        """
        if self.batch_first:
            x += self.pe[:, :x.size(1), :]
        else:
            x = x.transpose(0, 1)
            x += self.pe[:, :x.size(0), :]
            x = x.transpose(0, 1)
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    """Encoder-Decoder Transformer model."""

    def __init__(self, n_token: int, d_model: int, n_hid: int,
                 encoder_layers: int = 6, decoder_layers: int = 6,
                 n_head: int = 8, dropout: float = 0.1,
                 batch_first: bool = True):
        super(TransformerModel, self).__init__(d_model=d_model, nhead=n_head,
                                               dim_feedforward=n_hid, num_encoder_layers=encoder_layers,
                                               num_decoder_layers=decoder_layers,
                                               activation=F.gelu, dropout=dropout, batch_first=batch_first)

        self.input_emb = nn.Embedding(n_token, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, dropout, batch_first=batch_first)
        self.to_out = nn.Sequential(nn.Linear(d_model, n_token),
                                    nn.LogSoftmax(dim=-1))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.input_emb(src)
        src = self.pos_encoding(src)
        tgt = self.input_emb(tgt)
        tgt = self.pos_encoding(tgt)
        output = super(TransformerModel, self).forward(src=src, tgt=tgt,
                                                       src_mask=src_mask, tgt_mask=tgt_mask,
                                                       src_key_padding_mask=src_key_padding_mask,
                                                       tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.to_out(output)
        return output

    def lookup(self, x: torch.Tensor):
        """
        Embedding反向lookup，使用余弦相似度。
        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len]
        """
        embeddings = self.input_emb.weight   # shape: [n_token, d_model]
        # 归一化
        x_norm = F.normalize(x, p=2, dim=-1)  # [batch_size, seq_len, d_model]
        embeddings_norm = F.normalize(
            embeddings, p=2, dim=-1)  # [n_token, d_model]

        # 计算余弦相似度
        # [batch_size, seq_len, n_token]
        cos_sim = torch.matmul(x_norm, embeddings_norm.transpose(0, 1))

        # 找到相似度最高的token ids
        _, top_indices = cos_sim.topk(1, dim=-1)  # [batch_size, seq_len, 1]

        return top_indices.squeeze(-1)
        ...

    def generate(self, src: torch.Tensor, generate_len: int, src_mask: torch.Tensor, *,
                 src_key_padding_mask=None) -> torch.Tensor:
        """
        Generate a sequence of length generate_len. Greedy decoding is used.
        Args:
            src: torch.Tensor, shape [batch_size, seq_len]
            generate_len: int, the length of the generated sequence
            src_mask: torch.Tensor, shape [seq_len, seq_len]
            src_key_padding_mask: torch.Tensor, shape [batch_size, seq_len]
        Return:
            torch.Tensor, shape [batch_size, seq_len + generate_len]
        """
        generated = src
        for _ in range(generate_len):
            output = super(TransformerModel, self).forward(src=generated, tgt=generated,
                                                           src_mask=src_mask, tgt_mask=None,
                                                           src_key_padding_mask=src_key_padding_mask,
                                                           tgt_key_padding_mask=None)
            output = self.to_out(output)
            # 取最后一个时间步的预测结果，用于下一次的生成
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat(
                [generated, next_token.unsqueeze(-1)], dim=-1)

        return generated

    def beam_search(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, beam_size: int = 3):
        """
        用beam search生成文本。
        src: 输入序列的词嵌入表示
        src_mask: 输入序列的mask
        max_len: 生成序列的最大长度
        beam_size: beam大小
        """
        import numpy as np
        # 初始化beam
        beams = [src]  # 当前beam中的每个元素都是一个已生成的序列
        scores = [0]  # 每个beam元素的分数

        for _ in range(max_len):
            new_beams = []
            new_scores = []

            for beam_idx, beam in enumerate(beams):
                # 对当前beam进行预测
                output = self.forward(beam, beam, src_mask, None)
                output = self.to_out(
                    output[:, -1, :]).squeeze(1)  # 使用最后一个token的输出
                topk_scores, topk_indices = output.topk(
                    beam_size, dim=-1)  # 得到topk

                # 遍历topk结果，更新beam和分数
                for k in range(beam_size):
                    new_beam = torch.cat(
                        [beam, topk_indices[:, k:k+1]], dim=-1)
                    new_score = scores[beam_idx] + topk_scores[:, k].item()

                    new_beams.append(new_beam)
                    new_scores.append(new_score)

            # 从新beams中选择分数最高的beam_size个
            top_beams_indices = np.argsort(new_scores)[-beam_size:]
            beams = [new_beams[idx] for idx in top_beams_indices]
            scores = [new_scores[idx] for idx in top_beams_indices]

            # 如果所有beam都结束了，则停止
            all_done = all(beam[:, -1] == self.eos_token for beam in beams)
            if all_done:
                break

        # 选择分数最高的beam
        best_beam_index = scores.index(max(scores))
        best_beam = beams[best_beam_index]
        return best_beam
