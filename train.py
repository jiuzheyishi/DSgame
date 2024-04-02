import torch
from torch import nn as nn
from config import *
from dataset import TextDataset
from torch.utils.data import DataLoader
from torch import optim
import pickle as pkl
from tqdm import tqdm
import time
import model.models as models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 64

with open(WORD_IDX_PATH, "rb") as f:
    w2i = pkl.load(f)
train_iter = DataLoader(TextDataset(TRAIN_FLAG, w2i),
                        shuffle=True, batch_size=BATCH_SIZE)
print(next(iter(train_iter)))


def Train(net: nn.Module, lr=LEARNING_RATE):
    """训练序列到序列模型。"""

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

        # 将数据转换为成batch的Tensor
    with open(WORD_IDX_PATH, "rb") as f:
        w2i = pkl.load(f)
    # win上num_worker好像会出问题?
    train_iter = DataLoader(TextDataset(TRAIN_FLAG, w2i),
                            shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
    val_iter = DataLoader(TextDataset(VAL_FLAG, w2i),
                          shuffle=False, batch_size=BATCH_SIZE, num_workers=4)
    test_iter = DataLoader(TextDataset(TEST_FLAG, w2i),
                           shuffle=False, batch_size=1)

    net.apply(xavier_init_weights)
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss = models.MaskedSoftmaxCELoss()

    # 验证集loss降到10000以下时开始保存每轮更低的参数
    min_loss = 10000
    for epoch in range(EPOCHS):
        train_loss = []
        val_loss = []

        net.train()
        for batch in tqdm(train_iter):
            (enc_X, enc_x_l), (dec_x, dec_x_l), (y, y_l) = [
                (x[0].to(DEVICE), x[1].to(DEVICE)) for x in batch]

            pred, _ = net(enc_X, dec_x, enc_x_l)
            l = loss(pred, y, y_l).sum()
            l.backward()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                train_loss.append(l.item())

        # 释放显存
        torch.cuda.empty_cache()

        net.eval()
        with torch.no_grad():
            for batch in tqdm(val_iter):
                (enc_X, enc_x_l), (dec_x, dec_x_l), (y, y_l) = [
                    (x[0].to(DEVICE), x[1].to(DEVICE)) for x in batch]
                pred, _ = net(enc_X, dec_x, enc_x_l)
                l = loss(pred, y, y_l).sum()
                val_loss.append(l.item())

        # 保存模型参数，秒级时间戳保证唯一性
        if (sum(val_loss) < min_loss):
            min_loss = sum(val_loss)
            torch.save(net.state_dict(), PARAM_DIR +
                       str(int(time.time()))+"_GRU.param")
            print(f"saved net with val_loss:{min_loss}")
        print(f"{epoch+1}: train_loss:{sum(train_loss)};val_loss:{sum(val_loss)}")
