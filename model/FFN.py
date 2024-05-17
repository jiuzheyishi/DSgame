"""
Position-wise Feed-Forward Network (FFN)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout_prob: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        d_model: number of features in the input
        d_ff: number of features in the hidden layer
        dropout_prob: probability of an element to be zeroed
        is_gated: whether the hidden layer is gated
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        x = self.layer2(x)
        return x
