# -*- coding: utf-8 -*-
import os

# 特殊符号
PAD_WORD = '<pad>'
PAD_IDX = 0
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_NUM = 0
UNK_NUM = 1
BOS_NUM = 2
EOS_NUM = 3
# 读取数据时的标志
TRAIN_FLAG = 0
VAL_FLAG = 1
TEST_FLAG = 2

PARA_DIR = './paras'
DATA_DIR = './datasets'
VOCAB_FREQ_PATH = os.path.join(DATA_DIR, "vocab_cnt.pkl")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
WORD_IDX_PATH = os.path.join(DATA_DIR, "word2idx.pkl")
IDX_WORD_PATH = os.path.join(DATA_DIR, "idx2word.pkl")


# 词典大小,参考BERT的词典大小
VOCAB_SIZE = 30522
# 最长原文序列长度
MAX_SOURCE_LEN = 2193
# 最长摘要序列长度
MAX_SUMMARY_LEN = 587

# 限定序列长度（长于此长度做切割，短于此长度做padding）
SOURCE_THRESHOLD = 1000
SUMMARY_THRESHOLD = 60


# 数据清理规则 由于有通配符所以不要用r"xxx"的形式
PATTERNS_ONCE = [
    "by .*? published :.*?\. \| \..*? [0-9]+ \. ",
    "by \. [^\\n\\t]*? \. ",
    "-lrb- cnn -rrb- -- ",
    "\t(.*?-lrb- .*? -rrb- -- )",
    "updated : \. .*? \. ",
    "published : \. .*? \. \| \. "
]
PATTERNS_ANY = [
    r"``|''"
]
