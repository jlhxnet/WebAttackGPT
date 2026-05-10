# ===========================================
# loadmodel.py
# 加载模型和词表
# ===========================================

import torch
import os
from common import *
from gpt import GPT

def load_model():
    # 1. 加载词表
    vocab = torch.load(vocab_path, map_location=device, weights_only=False)
    # 2. 创建模型实例
    model = GPT(vocab_size=len(vocab), max_seq_len=max_seq_len).to(device)
    # 3. 加载训练好的参数
    model_state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(model_state_dict)
    # 4. 切换评估模式
    model.eval()
    
    print(f"✅ 模型&词表加载成功 | 设备：{device}")
    return model, vocab