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

## 2 模型结构 Seq2Seq+Attention

- Embedding: 将输入的单词转换为向量(torch)
- Encoder: 使用序列模型对输入的文本进行编码,得到每个时刻的隐向量和最后一个时刻的结果向量
- Decoder: 每一步解码,使用输出隐向量