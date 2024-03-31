import json
import torch
import re
import os
import pickle as pkl
from collections import Counter
import torch.nn as nn

DATA_DIR = './datasets/new_data'
EMBED_DIM = 256


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbedding, self).__init__()
        # 初始化词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_sequences):
        # 将输入序列（索引形式）转换为嵌入表示
        embedded_sequences = self.embedding(input_sequences)
        return embedded_sequences


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    def match(name): return bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def bulid_vocab_counter(data_dir):
    split_dir = os.path.join(data_dir, "train")
    n_data = _count_data(split_dir)
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

    with open(os.path.join(data_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)


def build_vocab(data_dir):
    # 假设你已经有了词汇表的计数器 vocab_counter
    with open(os.path.join(data_dir, "vocab_cnt.pkl"), 'rb') as vocab_file:
        vocab_counter = pkl.load(vocab_file)

    # 根据计数器创建词汇表，单词按出现频率排序，并映射到索引
    vocab = [word for word, count in vocab_counter.most_common()]
    vocab_dict = {word: i for i, word in enumerate(vocab)}

    with open(os.path.join(data_dir, "vocab.pkl"), 'wb') as f:
        pkl.dump(vocab_dict, f)

    return vocab, vocab_dict


def create_embedding_layer(vocab_dict, embedding_dim, padding_idx):
    # 使用词汇表创建嵌入层
    num_embeddings = len(vocab_dict)  # 词汇表大小
    embedding = nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx)
    # 可以加载预训练的词向量，如果需要的话
    # embedding.weight.data = pre_trained_embedding

    return embedding


# 示例使用
bulid_vocab_counter(DATA_DIR)
vocab, vocab_dict = build_vocab(DATA_DIR)

# 创建嵌入层
embedding_layer = create_embedding_layer(
    vocab_dict, embedding_dim=EMBED_DIM, padding_idx=0)
# 现在你可以使用embedding_layer来将单词转换为嵌入向量
# 例如，获取单词 "example" 的嵌入向量
word = "example"
if word in vocab_dict:
    word_idx = torch.tensor(vocab_dict[word])
    embedded_word = embedding_layer(word_idx)  # 返回一个嵌入向量
    print(embedded_word)
else:
    print(f"Word '{word}' not found in vocabulary.")
