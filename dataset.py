#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# @Time    : 2025/3/30 00:37
# @Author  : huangfujue
# @File    : dataset.py
# @Date    : 2025/3/30 
"""
模块说明
"""
import json

import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------------------------------------------
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):

                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length', # 不够的需要填充到这个长度
            truncation=True, # 是否超出长度后截断
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 根据前句子补全最后一个字
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        loss_mask = torch.as_tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

