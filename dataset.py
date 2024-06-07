"""
@brief: 生成TensorDataset
"""
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_FLAG, VAL_FLAG, TEST_FLAG, DATA_DIR, \
    UNK_NUM, EOS_NUM, BOS_NUM, SOURCE_THRESHOLD, SUMMARY_THRESHOLD
from utils import count_num_files, read_json2list, padding_seq


class Tokenizer:
    '''分词器'''

    def __init__(self, word2id: dict, id2word: dict):
        self.word2id = word2id
        self.id2word = id2word

    def encode(self, text: str):
        '''编码'''
        return [self.word2id[word] if word in self.word2id.keys()
                else UNK_NUM for word in text]

    def decode(self, tokens: list):
        '''解码'''
        return [self.id2word[token] for token in tokens]


class TextDataset(Dataset):
    '''生成TensorDataset'''

    def __init__(self, flag, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.path = DATA_DIR
        self.flag = flag
        if flag == TRAIN_FLAG:
            self.path = os.path.join(self.path, "new_train")
        elif flag == VAL_FLAG:
            self.path = os.path.join(self.path, "new_val")
        elif flag == TEST_FLAG:
            self.path = os.path.join(self.path, "new_test")
        else:
            raise ValueError(
                f"Invalid flag value: {flag}. Expected TRAIN_FLAG, VAL_FLAG, or TEST_FLAG.")

    def __len__(self):
        return count_num_files(self.path)

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.LongTensor, int],
                                               Tuple[torch.LongTensor, int],
                                               Tuple[torch.LongTensor, int]]:
        """
        @return: (enc_x, enc_x_l), (dec_x, dec_x_l), (y, y_l)
                编码器输入，编码器输入有效长度，解码器输入，解码器输入有效长度，标签，标签有效长度
        """
        source = read_json2list(self.path, index)
        summary = read_json2list(self.path, index, True)
        # 处理summary中奇怪的问题 ?
        # summary = [i for i in summary if (i != '' and i != ' ')]
        # print(summary)
        enc_x = self.tokenizer.encode(source)
        # padding
        enc_x, enc_x_l = padding_seq(enc_x, SOURCE_THRESHOLD)

        if self.flag != TEST_FLAG:
            dec_x = self.tokenizer.encode(summary)
            # decoder输入前面加上BOS、decoder的label最后加上EOS
            y = list(dec_x)
            y.append(EOS_NUM)
            y, y_l = padding_seq(y, SUMMARY_THRESHOLD)
            dec_x.insert(0, BOS_NUM)
            dec_x, dec_x_l = padding_seq(dec_x, SUMMARY_THRESHOLD)
        if self.flag == TEST_FLAG:
            return (torch.LongTensor(enc_x), enc_x_l), None, None
        return (torch.LongTensor(enc_x), enc_x_l), \
            (torch.LongTensor(dec_x), dec_x_l), \
            (torch.LongTensor(y), y_l)


if __name__ == "__main__":
    import pickle as pkl
    batch_size = 32
    with open(os.path.join(DATA_DIR, "word2idx.pkl"), "rb") as f:
        word2idx = pkl.load(f)
    with open(os.path.join(DATA_DIR, "idx2word.pkl"), "rb") as f:
        idx2word = pkl.load(f)
    tokenizer = Tokenizer(word2idx, idx2word)
    train_dataset = TextDataset(TRAIN_FLAG, tokenizer=tokenizer)
    val_dataset = TextDataset(VAL_FLAG, tokenizer=tokenizer)
    test_dataset = TextDataset(TEST_FLAG, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_loader), len(val_loader), len(test_loader))
    (enc_x, enc_x_l) = next(iter(train_loader))[0]
    print(enc_x.shape)
