"""
@brief: This module contains the implementation of multi-head attention.

"""

import math
from typing import Optional, List

import torch
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    """This module does a linear transformation and splits the vector into given number of heads for multi-head attention.
        This is used to transform key, query, and value vectors."""

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k  # number of dimensions in each head

    def forward(self, x: torch.Tensor):
        """
        @brief: apply linear transformation to the last dimension and split it into heads
        @input x: shape: [seq_len, batch_size, d_model] or [batch_size, d_model]
        @output: shape: [seq_len, batch_size, heads, d_k] or [batch_size, heads, d_k]
        """
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    """
    This computes scaled multi-headed attention for given query , key and value vectors.
    math: $$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
    In simple terms, it finds keys that matches the query, and gets the values of those keys.

    It uses dot-product of query and key as the indicator of how matching they are. Before taking the softmax the dot-products are scaled by  $\frac{1}{\sqrt{d_k}}$ . This is done to avoid large dot-product values causing softmax to give very small gradients when d_k  is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        heads: number of heads
        d_model : number of features in the query , key and value vectors.
        """
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.query = PrepareForMultiHeadAttention(
            d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(
            d_model, heads, self.d_k, True)
        # softmax is calculated along the axis of the sequence (or time)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None  # stores the attention scores for logging or other purposes
        self.output = nn.Linear(d_model, d_model)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        Calculate QK^T or $S_{ijbh}=\sum_{d} Q_{ibhd}K_{jbhd}$
        This method can be overridden for other variations like relative attention.
        """
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        mask has shape [seq_len_q, seq_len_k, batch_size] , where first dimension is the query dimension. 
        If the query dimension is equal to 1 it will be broadcasted.
        @input: shape:  [seq_len_q, seq_len_k, batch_size]
        @output: shape: [seq_len_q, seq_len_k, batch_size, heads(boradcasted)]
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        # same mask for all heads
        return mask.unsqueeze(-1)

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        query: shape: [seq_len, batch_size, d_model]
        key: shape: [seq_len, batch_size, d_model]
        value: shape: [seq_len, batch_size, d_model]
        mask: shape: [seq_len, seq_len, batch_size]
        """
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores = scores * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        # shape: [seq_len, batch_size, heads,
        attn = self.dropout(attn)
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, self.d_model)
        return self.output(x)
