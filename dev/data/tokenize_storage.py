#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# from utils.file_utils import load_base_config
from transformers import AutoTokenizer
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm

# config = load_base_config("conf/config.yaml")
data_file = "tang/tang.txt"

# 读取数据
with open(data_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 去除空行和多余的空白符
chinese_data = [line.strip() for line in lines if line.strip()]
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="../model/bert-base-chinese")
# 添加结束标记 <|endoftext|> 到分词器
tokenizer.add_special_tokens(special_tokens_dict={'eos_token': '<|endoftext|>'})
# 设置每个片段的最大 token 数
context_length = 128


def tokenize(doc):
    # 为每个文档添加 <|endoftext|> 标记
    text_with_eot = doc + tokenizer.eos_token  # 这里会使用已经添加的结束标记
    tokens = tokenizer(
        text_with_eot,
        truncation=True,
        max_length=context_length,
        return_tensors="np"
    )["input_ids"]

    tokens_np = np.array(tokens.flatten(), dtype=np.uint16)

    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token 字典对于 uint16 来说太大"
    return tokens_np


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


shard_size = int(1e6)  # 每个分片的 token 数
output_dir = "tang_tokenized_data"
os.makedirs(output_dir, exist_ok=True)
# 假设中文数据是一个列表，每个元素是一个文档文本
chinese_data = chinese_data
# chinese_data = ["文档1的内容", "文档2的内容", "文档3的内容"]  # 示例数据
nprocs = max(1, os.cpu_count() // 2)
# 标记所有文档并写入输出分片，每个分片shard_size令牌（最后一个分片有剩余）
# with mp.Pool(nprocs) as pool: # 多线程
shard_index = 0
# preallocate buffer 以保存当前分片
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None


# 判断当前分片中是否有足够的空间用于新token？
# for tokens in pool.imap(tokenize, chinese_data, chunksize=16):
for token in chinese_data:
    tokens = tokenize(token)
    if token_count + len(tokens) < shard_size:
        # 只需将 Token 附加到当前分片
        all_tokens_np[token_count:token_count + len(tokens)] = tokens
        token_count += len(tokens)

        # 更新进度条
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        # 写入当前分片并启动新分片
        filename = os.path.join(output_dir, f"tang_shard_{shard_index:06d}.npy")

        # 将文档拆分为适合此分片的任何内容，其余的转到下一个
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None

        # 使用当前文档的剩余部分填充下一个分片
        all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

# 将任何剩余的 Token 写入最后一个分片
if token_count != 0:
    filename = os.path.join(output_dir, f"tang_shard_{shard_index:06d}.npy")
    write_datafile(filename, all_tokens_np[:token_count])