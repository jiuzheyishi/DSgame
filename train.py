'''
train process, still have some problems
'''

import pickle as pkl
from typing import Tuple, Callable

import torch
from torch import nn as nn
from torch import optim, Tensor

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextDataset
from .model.Transformer import EncoderDecoder, Encoder, Decoder
from config import *
from utils import get_rouge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 64

LossFn = Callable[[Tensor, Tensor], Tensor]


def train_step(
    model: torch.nn.Module,
    enc_x: Tensor, dec_x: Tensor, y,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    y_pred = model(enc_x, dec_x)
    loss = loss_fn(y_pred, y)
    metric = get_rouge(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item(), metric


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    enc_x: Tensor, dec_x: Tensor, y,
    loss_fn: LossFn
) -> Tuple[float, float]:
    model.eval()
    y_pred = model(enc_x, dec_x)
    loss = loss_fn(y_pred, y)
    metric = get_rouge(y_pred, y)
    return loss.item(), metric


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device = DEVICE,
    loss_fn: LossFn = nn.CrossEntropyLoss(),
    max_epochs: int = EPOCHS,
    early_stop: int = 3,
    print_every: int = 3,
    verbose: bool = True
):
    for epoch in range(max_epochs):
        for i, (())
