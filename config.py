import os

# 特殊符号
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_NUM = 0
UNK_NUM = 1
BOS_NUM = 2
EOS_NUM = 3
# 读取数据时的标志
TRAIN_FALG = 0
VAL_FALG = 1
TEST_FALG = 2

PARA_DIR = './paras'
DATA_DIR = './datasets/new_data'
VOCAB_FREQ_PATH = os.path.join(DATA_DIR, "vocab_cnt.pkl")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
WORD_IDX_PATH = os.path.join(DATA_DIR, "word2idx.pkl")
IDX_WORD_PATH = os.path.join(DATA_DIR, "idx2word.pkl")

VOCAB_SIZE = 50_000

# 词典大小(拉满就不会出现UNK),注意输入至网络时要加4（还有四个特殊字符）
VOCAB_SIZE = 10000
# 最长原文序列长度
MAX_SOURCE_LEN = 2193
# 最长摘要序列长度
MAX_SUMMARY_LEN = 587

# 限定序列长度（长于此长度做切割，短于此长度做padding）
SOURCE_THRESHOLD = 1800
SUMMARY_THRESHOLD = 550


# 训练参数
BATCH_SIZE = 64
EPOCHES = 10
