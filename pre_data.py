"""
@brief: 数据预处理,包括数据清洗、划分验证集、保存为json文件
@author: portrilin
@date: 2024.5.16
"""

import json
import os
import re
from tqdm import tqdm
from config import DATA_DIR, TRAIN_FLAG, VAL_FLAG, \
    TEST_FLAG, PATTERNS_ONCE, PATTERNS_ANY


############################### Just run for one time! ###############################

def preprocess(train_path=os.path.join(DATA_DIR, "train_dataset.csv"),
               test_path=os.path.join(DATA_DIR, "test_dataset.csv")):
    '''
    清理数据、划分验证集后重新保存至新文件
    '''

    # 数据清洗
    def clean_data(data):
        print("数据清洗开始=========================================")

        clean_data = []
        for _, d in tqdm(enumerate(data)):
            res = d
            for pat in PATTERNS_ONCE:
                # 之后修改
                if "\t" in pat:
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
    def save2json(data, mode):
        if mode == TEST_FLAG:
            for i, item in enumerate(data):
                source = item.split('\t')[1].strip('\n')
                if source != '':
                    dict_data = {"text": source,
                                 "summary": 'no summary'}  # 测试集没有参考摘要
                    with open(new_test_path+str(i)+'.json', 'w+', encoding='utf-8') as f:
                        f.write(json.dumps(dict_data, ensure_ascii=False))
        else:
            for i, item in enumerate(data):
                if len(item.split('\t')) == 3:
                    source_seg = item.split("\t")[1]
                    traget_seg = item.split("\t")[2].strip('\n')

                    if source_seg and traget_seg != '':
                        dict_data = {"text": source_seg, "summary": traget_seg}
                        path = new_train_path
                        if mode == 1:
                            path = new_val_path
                        with open(path+str(i)+'.json', 'w+', encoding='utf-8') as f:
                            f.write(json.dumps(dict_data, ensure_ascii=False))
                else:
                    print('problem data:', item)

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data_all = f.readlines()

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    # 数据清洗
    train_data_all = clean_data(train_data_all)
    test_data = clean_data(test_data)

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

    save2json(train_data, TRAIN_FLAG)
    save2json(val_data, VAL_FLAG)
    save2json(test_data, TEST_FLAG)


if __name__ == "__main__":
    preprocess()
