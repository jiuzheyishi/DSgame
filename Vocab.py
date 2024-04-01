import json
import torch
import re
import os
import pickle as pkl
from collections import Counter
import torch.nn as nn

from config import *  # const & paras


def count_num_files(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    def match(name): return bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def bulid_vocab_counter(self, data_dir):
    '''
    统计所有词汇，建立词频表
    '''
    split_dir = os.path.join(data_dir, "train")
    n_data = count_num_files(split_dir)
    vocab_counter = Counter()
    for i in range(n_data):
        js = json.load(
            open(os.path.join(split_dir, '{}.json'.format(i)), encoding='utf-8'))

        summary = js['summary']
        summary_text = ' '.join(summary).strip()
        summary_word_list = summary_text.strip().split(' ')

        review = js['source']
        review_text = ' '.join(review).strip()
        review_word_list = review_text.strip().split(' ')

        all_tokens = summary_word_list + review_word_list
        vocab_counter.update([t for t in all_tokens if t != ""])

    with open(os.path.join(data_dir, VOCAB_FREQ_PATH),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)


def build_vocab(data_dir):
    # 假设你已经有了词汇表的计数器 vocab_counter
    with open(os.path.join(data_dir, VOCAB_FREQ_PATH), 'rb') as vocab_file:
        vocab_counter = pkl.load(vocab_file)

    # 根据计数器创建词汇表，单词按出现频率排序，并映射到索引
    vocab = [word for word, count in vocab_counter.most_common()]
    vocab_dict = {word: i for i, word in enumerate(vocab)}

    with open(os.path.join(data_dir, "vocab.pkl"), 'wb') as f:
        pkl.dump(vocab_dict, f)

    return vocab, vocab_dict


def MakeVocab(vocab_size=VOCAB_SIZE):
    '''
    建立词典,通过vocab_size设置字典大小,将常用词设置到字典即可,其他生僻词汇用'<unk>'表示
    '''
    with open(VOCAB_PATH, "rb") as f:
        wc = pkl.load(f)
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w

    with open(WORD_IDX_PATH, "wb") as f:
        pkl.dump(word2idx, f)
    with open(IDX_WORD_PATH, "wb") as f:
        pkl.dump(idx2word, f)


def GetNumOfLongestSeq():
    '''
    找到最长的seq长度,用于padding
    '''
    def _findInFolders(path, length):
        max_len = 0
        for i in range(length):
            js_data = json.load(
                open(os.path.join(path, f"{i}.json"), encoding="utf-8"))
            l_data = js_data["summary"].split(" ")
            l = len(l_data)
            if (max_len < len(l_data)):
                max_len = l
        return max_len

    train_path = os.path.join(DATA_DIR, "train/")
    val_path = os.path.join(DATA_DIR, "val/")
    test_path = os.path.join(DATA_DIR, "test/")

    train_length = count_num_files(train_path)
    val_length = count_num_files(val_path)
    test_length = count_num_files(test_path)

    return max(
        _findInFolders(train_path, train_length),
        _findInFolders(val_path, val_length),
        _findInFolders(test_path, test_length))
