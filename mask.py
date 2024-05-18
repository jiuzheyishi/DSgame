
import torch
from torch import nn as nn
from typing import Tuple
from config import PAD_IDX
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence. 
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    often used in Decoder Layer.
    """
    mask = (torch.triu(torch.ones((size, size), device=DEVICE))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = PAD_IDX) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == "__main__":
    mask = generate_square_subsequent_mask(3)
    print(mask)
