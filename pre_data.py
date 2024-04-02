import json
import random
import os
from config import *
from tqdm import tqdm
import re
# 数据清洗


############################### Just run for one time! ###############################


def Preprocess(train_path=os.path.join(DATA_DIR, "train_dataset.csv"), test_path=os.path.join(DATA_DIR, "test_dataset.csv")):
    '''
    清理数据、划分验证集后重新保存至新文件
    '''

    # 数据清洗
    def _cleanData(data):
        print("数据清洗开始=========================================")

        clean_data = []
        for i, d in tqdm(enumerate(data)):
            res = d
            # if i == 1940:
            #     print(d)
            for pat in PATTERNS_ONCE:
                # 之后修改
                if ("\t" in pat):
                    res = re.sub(pat, "\t", res, 1)
                else:
                    res = re.sub(pat, "", res, 1)
                ####################################
            for pat in PATTERNS_ANY:
                res = re.sub(pat, "", res)

            clean_data.append(res)

        print("数据清洗完毕=========================================")
        return clean_data

    # 将处理后的数据保存为json文件
    def _save2Json(data, mode):

        if mode == TEST_FALG:
            for i in range(len(data)):
                source = data[i].split('\t')[1].strip('\n')
                if source != '':
                    dict_data = {"text": source,
                                 "summary": 'no summary'}  # 测试集没有参考摘要

                    with open(new_test_path+str(i)+'.json', 'w+', encoding='utf-8') as f:
                        f.write(json.dumps(dict_data, ensure_ascii=False))

        else:
            for i in range(len(data)):

                if len(data[i].split('\t')) == 3:
                    source_seg = data[i].split("\t")[1]
                    traget_seg = data[i].split("\t")[2].strip('\n')

                    if source_seg and traget_seg != '':
                        dict_data = {"text": source_seg, "summary": traget_seg}
                        path = new_train_path
                        if mode == 1:
                            path = new_val_path
                        with open(path+str(i)+'.json', 'w+', encoding='utf-8') as f:
                            f.write(json.dumps(dict_data, ensure_ascii=False))
                else:
                    print('problem data:', data[i])

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data_all = f.readlines()

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    # 数据清洗
    train_data_all = _cleanData(train_data_all)
    test_data = _cleanData(test_data)

    # 设置新文件路径
    new_train_path = os.path.join(DATA_DIR, "new_train/")
    new_val_path = os.path.join(DATA_DIR, "new_val/")
    new_test_path = os.path.join(DATA_DIR, "new_test/")

    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)

    if not os.path.exists(new_val_path):
        os.makedirs(new_val_path)

    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)

    # 把训练集重新划分为训练子集和验证子集,保证验证集上loss最小的模型,预测测试集
    train_data = train_data_all[:8000]
    val_data = train_data_all[8000:]

    _save2Json(train_data, TRAIN_FALG)
    _save2Json(val_data, VAL_FALG)
    _save2Json(test_data, TEST_FALG)


Preprocess()
