"""
@brief: some utils functions
"""
import json
import re
import os

from rouge import Rouge
from config import DATA_DIR, EOS_NUM, PAD_NUM


def get_rouge(pred, label):
    '''获取ROUGR-L值'''
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred, label)
    rouge_l_f1 = 0
    rouge_l_p = 0
    rouge_l_r = 0
    for d in rouge_score:
        rouge_l_f1 += d["rouge-l"]["f"]
        rouge_l_p += d["rouge-l"]["p"]
        rouge_l_r += d["rouge-l"]["r"]

    return rouge_l_f1 / len(rouge_score)


def count_num_files(path: str) -> int:
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')

    def match(name):
        return bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def get_num_longest_seq() -> int:
    '''
    找到最长的seq长度,用于padding
    '''
    def find_max_len(path, length):
        '''找到文件夹中最长的seq长度'''
        max_len = 0
        for i in range(length):
            js_data = json.load(
                open(os.path.join(path, f"{i}.json"), encoding="utf-8"))
            l_data = js_data["summary"].split(" ")
            l = len(l_data)
            if max_len < len(l_data):
                max_len = l
        return max_len

    train_path = os.path.join(DATA_DIR, "new_train/")
    val_path = os.path.join(DATA_DIR, "new_val/")
    test_path = os.path.join(DATA_DIR, "new_test/")

    train_length = count_num_files(train_path)
    val_length = count_num_files(val_path)
    test_length = count_num_files(test_path)

    return max(
        find_max_len(train_path, train_length),
        find_max_len(val_path, val_length),
        find_max_len(test_path, test_length))


def padding_seq(line, threshold: int):
    """
    @brief: padding序列到固定长度
    @param line: 输入序列,threshold:最大长度
    @return: 返回padded list, min(原始长度,threshold)
    """
    p_len = len(line)
    if p_len > threshold:
        if EOS_NUM in line:
            line[threshold-1] = EOS_NUM
        return line[:threshold], threshold
    return line + [PAD_NUM] * (threshold - len(line)), p_len


def read_json2list(dirname: str, i: str, label: bool = False):
    '''
    @brief: 读取dirname目录下的json文件,
    @param dirname: 文件目录, i: 第i个文件名, label: True for summary, False for text
    @return: text或summary的word list
    '''
    js_data = json.load(
        open(os.path.join(dirname, f"{i}.json"), encoding="utf-8"))
    if label:
        return js_data["summary"].split(" ")
    return js_data["text"].split(" ")
