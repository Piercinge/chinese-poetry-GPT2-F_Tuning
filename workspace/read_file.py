#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# import numpy as np
#
# data_file = "../dataset/tang.npz"
# data = np.load(data_file, allow_pickle=True)
# p = data["data"][0]
# for i in p:
#     print(i)
# print(data["ix2word"])
import numpy as np

# 加载数据
data_file = "../dataset/tang.npz"
data = np.load(data_file, allow_pickle=True)
print(data)

# 获取需要映射的索引序列和索引到单词的映射字典
data_sequence = data["data"][11533]  # 数据中的索引序列
ix2word = data["ix2word"].item()  # 提取字典

# 将索引映射为实际单词
mapped_sequence = [ix2word.get(ix, "<UNK>") for ix in data_sequence]

# 拼接为句子并输出（假设句子以特定标记结束，例如句号）
output = ""
for word in mapped_sequence:
    output += word
    if word in {"。", "！", "？", "<START>"}:  # 判断是否是句子的结束标点
        print(output)
        output = ""

# 如果输出中有剩余内容（没有以句号结束）
if output:
    print(output)

