import math

import torch
import torch.nn as nn

from .PositionEncoding import get_positional_encoding
from .MultiheadAttention import MultiheadAttention
from .FFN import FeedForward


class EmbeddingsWithPositionalEncoding(nn.Module):
    """fixed positional encoding"""

    def __init__(self,  d_model: int, n_vocab: int, max_len: int = 5000):
        """
        n_vocab: number of words in the vocabulary
        d_model: number of features in the query, key, and value vectors
        max_len: maximum length of the input sequence
        """
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(n_vocab, d_model)
        self.register_buffer('positional_encoding',
                             get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]].requires_grad_(False)
        x = self.embeddings(x)*math.sqrt(self.d_model)
        x = x + pe
        return x


class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    """learned positional encoding"""

    def __init__(self,  d_model: int, n_vocab: int, max_len: int = 5000):
        """
        n_vocab: number of words in the vocabulary
        d_model: number of features in the query, key, and value vectors
        max_len: maximum length of the input sequence
        """
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(n_vocab, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]]
        x = self.embeddings(x)*math.sqrt(self.d_model)
        x = x + pe
        return x


class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiheadAttention,
                 src_attn: MultiheadAttention = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        d_model: token embedding size
        self_attn: self attention layer
        src_attn: source attention layer (when used in decoder)
        feed_forward: feed forward layer
        """

        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

        self.is_save_ff_input = False  # whether to save the input of feed forward layer

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)
        if src is not None:
            z = self.norm_src_attn(x)
            src_attn = self.src_attn(
                query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(src_attn)
        z = self.norm_ff(x)
        if self.is_save_ff_input:
            self.ff_input = z
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 layer: TransformerLayer,
                 n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self,
                 layer: TransformerLayer,
                 n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        """
        memory: encoder output

        """
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)


class Generator(nn.Module):
    """This predicts the tokens and gives the lof softmax of those. 
    You don't need this if you are using nn.CrossEntropyLoss ."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        return self.proj(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
