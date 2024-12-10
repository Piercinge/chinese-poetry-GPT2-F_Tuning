#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# 文件路径列表
input_files_paths = [
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\quantang.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\tang_e.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\chuci.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\huajianji.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\nanlan.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\cao.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\nantang.txt",
    r"E:\B_pythonProject\chinese-poetry-BERT-F_Tuning\dev\data\tang\ytang.txt",
]

# 初始化一个空列表来存储所有文件的内容
lines = []

# 循环读取每个文件的内容
for input_file_path in input_files_paths:
    with open(input_file_path, "r", encoding="utf-8") as f:
        lines.extend(f.readlines())

# 去除空行和多余的空白符
chinese_data = [line.strip() for line in lines if line.strip()]

# 初始化BERT分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="../../dev/model/bert-base-chinese")
tokenizer.add_special_tokens(special_tokens_dict={'eos_token': '<|endoftext|>'})

# 设置每个片段的最大 token 数
context_length = 128  # 数据集中一个完整的诗长度不会超过128

def tokenize(doc):
    text_with_eot = doc.replace("\n", "")  # 去掉换行符
    tokens = tokenizer(
        text_with_eot,
        truncation=True,
        max_length=context_length,
        return_tensors="np"
    )["input_ids"][0][1:-1]  # 去掉bert分词器默认的cls和sep

    tokens_np = np.array(tokens.flatten(), dtype=np.uint16)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token 字典对于 uint16 来说太大"
    return tokens_np

# 将所有 token 加入到一个大的 numpy 数组
all_tokens = []

for token in tqdm(chinese_data, desc="Tokenizing text"):
    tokens = tokenize(token)
    all_tokens.append(tokens)

# 将所有的tokens合并为一个大的np数组
all_tokens_np = np.concatenate(all_tokens)

# 划分为训练集和验证集 (90% 训练集，10% 验证集)
split_idx = int(len(all_tokens_np) * 0.9)

train_tokens = all_tokens_np[:split_idx]
val_tokens = all_tokens_np[split_idx:]

# 将训练集和验证集分别保存到文件
def save_tokens(filename, tokens):
    tokens.tofile(filename)

# 保存训练集和验证集
output_dir = "poem"
os.makedirs(output_dir, exist_ok=True)

train_filename = os.path.join(output_dir, "train.bin")
val_filename = os.path.join(output_dir, "val.bin")

save_tokens(train_filename, train_tokens)
save_tokens(val_filename, val_tokens)

print(f"token number: {len(all_tokens_np)}")
print(f"Training data saved to {train_filename}, train size: {len(train_tokens)}")
print(f"Validation data saved to {val_filename}, val size: {len(val_tokens)}")

# Training data saved to poem\train.bin, train size: 18305565
# Validation data saved to poem\val.bin, val size: 2033952
