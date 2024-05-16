# DS games
## Brief

- 深度学习导论 课程竞赛
- 组员: 
- 新闻摘要自动生成 
Automatic generation of news summaries
- 比赛网址: https://www.datafountain.cn/competitions/541
- Code format: PEP 8
## Data

- 数据集包含了新闻的标题和正文，以及对应的摘要。
- run一次`pre_data.py`可将每条训练数据以转化为json格式,run `Vocab.py`可生成词典


## 1 数据处理
本项目数据处理共分为部分：数据划分、词典生成、张量转换
+ 数据清洗与划分
  + 读取数据并清洗,去除固定pattern而无意义的数据 
  + 从原始训练集中划分出验证集
  + 将原始CSV文件转换为逐条文本的JSON文件
+ 词典生成  
+ 张量转换  

## 2 模型结构 TODO
本项目使用`pytorch`实现了模型基础结构、自定义损失函数、优化器以及模型训练、验证过程；  

1. GRU+RNN

2. Transformer+ 预训练模型 