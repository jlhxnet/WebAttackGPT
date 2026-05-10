# ============================================================
# poswisefeedforwardnet.py
# 位置前馈网络
# 1D卷积层
# 1D卷积层
# 前馈网络Dropout
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *

class PositionwiseFeedForwardNet(nn.Module):
    def __init__(self, d_embedding, d_ff, dropout):
        super().__init__()
        # 1D卷积层,输入d_embedding,输出d_ff,卷积核大小1
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 1D卷积层,输入d_ff,输出d_embedding,卷积核大小1
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
 
        # 前馈网络Dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, inputs):
        # 输入的形状为(batch_size, seq_len, d_embedding)，输出的形状为(batch_size, d_ff, seq_len)
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.dropout1(output)
        output = self.conv2(output)
        output = self.dropout2(output)
        # 维度还原：(batch, seq_len, d_emb)
        output = output.transpose(1, 2)
        return output