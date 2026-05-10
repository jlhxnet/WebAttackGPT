import numpy as np
import torch
import torch.nn as nn
from common import *

# 缩放点积注意力层：用于解码器的多头自注意力层，计算Q和K的点积，然后除以sqrt(d_k)，然后将结果和V相乘，得到注意力值
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,Q,K,V,attn_mask):
        # Q的维度是(batch_size, n_heads, seq_len, d_k)
        # K的维度是(batch_size, n_heads, seq_len, d_k)
        # transpose(-1,-2)的作用是将K的最后两个维度交换位置，K交换后变成(batch_size, n_heads, d_k, seq_len)
        # Q(batch_size, n_heads, seq_len, d_k)*K(batch_size, n_heads, d_k, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)
        # masked_fill作用是将被mask的位置填充为-1e9
        scores = scores.masked_fill(attn_mask, -1e9)
        # 计算softmax，dim=-1表示最后一个维度，得到(batch_size, n_heads, seq_len, seq_len个注意力权重)
        weights = nn.Softmax(dim=-1)(scores)
        # 计算注意力值，weights(batch_size, n_heads, seq_len, seq_len) * V(batch_size, n_heads, seq_len, d_v) -> (batch_size, n_heads, seq_len, d_v)
        context = torch.matmul(weights,V)
        return context, weights