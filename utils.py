from rouge import Rouge
import json
from config import *


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


def PaddingSeq(line, threshold):
    """填充文本序列,直接填充转换完的index列表"""
    p_len = len(line)
    if (p_len > threshold):
        if (EOS_NUM in line):
            line[threshold-1] = EOS_NUM
        return line[:threshold], threshold
    return line + [PAD_NUM] * (threshold - len(line)), p_len


def ReadJson2List(dir, i, label=False):
    '''读取单个json文件(一个样本),并按空格分割转换成列表'''
    js_data = json.load(open(os.path.join(dir, f"{i}.json"), encoding="utf-8"))
    if label:
        return js_data["summary"].split(" ")
    return js_data["text"].split(" ")
