"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import json
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
# -----------------------------------------------------------------------------


# 使用中文GPT-2模型
enc = AutoTokenizer.from_pretrained("bert-base-chinese") # 或者 "bert-base-chinese" 等
model = GPT2LMHeadModel.from_pretrained("gpt2")


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def render_example_chinese(example):
    """
    给定中文诗词的例子，返回处理过的 tokens 和 mask，保持原有结构
    - tokens (4xN)：包含上下文 + 结束部分的 token，每行是一个候选的完整句子
    - mask：用来指示哪部分是候选部分的掩码
    - label：正确的结束部分索引
    """
    ctx = example["ctx"]  # 上下文部分
    label = example["label"]  # 正确的结束部分索引
    endings = example["endings"]  # 结束部分候选

    # Tokenize 使用中文的tokenizer
    ctx_tokens = enc.encode(ctx, add_special_tokens=False)  # 对上下文进行编码
    tok_rows = []  # 用来存储每个候选完整句子的tokens
    mask_rows = []  # 对应的掩码，指示候选部分的区域

    # 为每个候选结束部分生成 tokens 和掩码
    for end in endings:
        end_tokens = enc.encode(end, add_special_tokens=False)  # 编码候选结束部分
        tok_rows.append(ctx_tokens + end_tokens)  # 上下文和结束部分拼接
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))  # 上下文部分为 0，结束部分为 1

    # 找到最大长度，以便对齐所有的句子
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)  # 初始化 tokens 张量，4 行（每个候选）最多 max_len 长度
    mask = torch.zeros((4, max_len), dtype=torch.long)  # 初始化 mask 张量

    # 填充 tokens 和 mask
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)  # 填充 tokens
        mask[i, :len(mask_row)] = torch.tensor(mask_row)  # 填充 mask

    # 组织数据
    data = {
        "label": label,  # 记录正确的候选索引
        "ctx_tokens": ctx_tokens,  # 上下文的 tokens
        "ending_tokens": [enc.encode(end, add_special_tokens=False) for end in endings]  # 所有结束部分的 tokens
    }

    return data, tokens, mask, label


def iterate_examples(split):
    # 假设你已经有一个中文诗词数据集文件
    data_file = f"./hellaswag/chinese_poetry_{split}.jsonl"
    print(data_file)
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate_chinese(model, device):
    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model)
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):  # 使用中文诗词数据集
        _, tokens, mask, label = render_example_chinese(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Compute logits and loss
        logits = model(tokens).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        masked_shift_losses = shift_losses * mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / mask.sum(dim=1)

        pred = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        print(f"{num_total} acc: {num_correct}/{num_total}={num_correct/num_total:.4f}")


if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
#     parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
#     args = parser.parse_args()
#     evaluate(args.model_type, args.device)
    iterate_examples("train")
