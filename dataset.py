from config import *
from utils import *
from Vocab import *
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    '''生成TensorDataset'''

    def __init__(self, flag, word2id: dict):
        self.word2id = word2id
        self.path = DATA_DIR
        self.flag = flag
        if (flag == TRAIN_FALG):
            self.path += "new_train"
        elif (flag == VAL_FALG):
            self.path += "new_val"
        elif (flag == TEST_FALG):
            self.path += "new_test"
        else:
            raise Exception(f"No this flag:{flag}")

    def __len__(self):
        return count_num_files(self.path)

    def __getitem__(self, index):
        source = ReadJson2List(self.path, index)
        summary = ReadJson2List(self.path, index, True)
        # 处理summary中奇怪的问题
        # summary = [i for i in summary if (i != '' and i != ' ')]
        # print(summary)
        enc_x = [self.word2id[word] if word in self.word2id.keys()
                 else UNK_NUM for word in source]
        # padding
        enc_x, enc_x_l = PaddingSeq(enc_x, SOURCE_THRESHOLD)

        if (self.flag != TEST_FALG):
            dec_x = [self.word2id[word] if word in self.word2id.keys()
                     else UNK_NUM for word in summary]
            # decoder输入前面加上BOS、decoder的label最后加上EOS
            y = list(dec_x)
            y.append(EOS_NUM)
            y, y_l = PaddingSeq(y, SUMMARY_THRESHOLD)

            dec_x.insert(0, BOS_NUM)
            dec_x, dec_x_l = PaddingSeq(dec_x, SUMMARY_THRESHOLD)
        if (self.flag == TEST_FALG):
            return (torch.LongTensor(enc_x), enc_x_l)
        # return ：编码器输入，编码器输入有效长度，解码器输入，解码器输入有效长度，标签，标签有效长度
        return (torch.LongTensor(enc_x), enc_x_l), (torch.LongTensor(dec_x), dec_x_l), (torch.LongTensor(y), y_l)
