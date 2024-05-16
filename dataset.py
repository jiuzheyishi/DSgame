"""
@brief: 生成TensorDataset
"""
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from config import TRAIN_FLAG, VAL_FLAG, TEST_FLAG, DATA_DIR, \
    UNK_NUM, EOS_NUM, BOS_NUM, SOURCE_THRESHOLD, SUMMARY_THRESHOLD
from utils import count_num_files, read_json2list, padding_seq


class TextDataset(Dataset):
    '''生成TensorDataset'''

    def __init__(self, flag, word2id: dict, id2word: dict):
        self.word2id = word2id
        self.id2word = id2word
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

    def __getitem__(self, index: int):
        """
        @return: (enc_x, enc_x_l), (dec_x, dec_x_l), (y, y_l)
        """
        source = read_json2list(self.path, index)
        summary = read_json2list(self.path, index, True)
        # 处理summary中奇怪的问题 ?
        # summary = [i for i in summary if (i != '' and i != ' ')]
        # print(summary)
        enc_x = [self.word2id[word] if word in self.word2id.keys()
                 else UNK_NUM for word in source]
        # padding
        enc_x, enc_x_l = padding_seq(enc_x, SOURCE_THRESHOLD)

        if self.flag != TEST_FLAG:
            dec_x = [self.word2id[word] if word in self.word2id.keys()
                     else UNK_NUM for word in summary]
            # decoder输入前面加上BOS、decoder的label最后加上EOS
            y = list(dec_x)
            y.append(EOS_NUM)
            y, y_l = padding_seq(y, SUMMARY_THRESHOLD)

            dec_x.insert(0, BOS_NUM)
            dec_x, dec_x_l = padding_seq(dec_x, SUMMARY_THRESHOLD)
        if self.flag == TEST_FLAG:
            return (torch.LongTensor(enc_x), enc_x_l)
        # return ：编码器输入，编码器输入有效长度，解码器输入，解码器输入有效长度，标签，标签有效长度
        return (torch.LongTensor(enc_x), enc_x_l), \
            (torch.LongTensor(dec_x), dec_x_l), \
            (torch.LongTensor(y), y_l)

    def inverse_transform(self, x: torch.Tensor):
        '''将数字序列转换为文本'''
        text = [self.id2word[i] for i in x]
        return text


if __name__ == "__main__":
    import pickle as pkl
    with open(os.path.join(DATA_DIR, "word2idx.pkl"), "rb") as f:
        word2idx = pkl.load(f)
    with open(os.path.join(DATA_DIR, "idx2word.pkl"), "rb") as f:
        idx2word = pkl.load(f)
    dataset = TextDataset(TRAIN_FLAG, word2idx, idx2word)
    enx0, enx_l0, decx0, decx_l0, y0, y_l0 = dataset[0]
    print(enx0)
