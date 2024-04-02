from rouge import Rouge
import json
from config import *
import re
import os


def GetRouge(pred, label):
    '''获取ROUGR-L值'''
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred, label)
    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]

    return (rouge_L_f1 / len(rouge_score))

    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))


def count_num_files(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    def match(name): return bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


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

    train_path = os.path.join(DATA_DIR, "new_train/")
    val_path = os.path.join(DATA_DIR, "new_val/")
    test_path = os.path.join(DATA_DIR, "new_test/")

    train_length = count_num_files(train_path)
    val_length = count_num_files(val_path)
    test_length = count_num_files(test_path)

    return max(
        _findInFolders(train_path, train_length),
        _findInFolders(val_path, val_length),
        _findInFolders(test_path, test_length))


def PaddingSeq(line, threshold):
    """
    @brief: padding序列到固定长度
    @param line: 输入序列,threshold:最大长度
    @return: 返回padded list, 原始长度/threshold
    """
    p_len = len(line)
    if (p_len > threshold):
        if (EOS_NUM in line):
            line[threshold-1] = EOS_NUM
        return line[:threshold], threshold
    return line + [PAD_NUM] * (threshold - len(line)), p_len


def ReadJson2List(dir, i, label=False):
    '''
    @brief: 读取dir目录下的json文件,
    @param dir: 文件目录, i: 文件名, label: True for summary, False for text
    @return: text或summary的word list
    '''
    js_data = json.load(open(os.path.join(dir, f"{i}.json"), encoding="utf-8"))
    if label:
        return js_data["summary"].split(" ")
    return js_data["text"].split(" ")
