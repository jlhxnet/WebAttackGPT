import torch
import torch.nn as nn
import numpy as np
from decoderlayer import DecoderLayer
from common import *

# 解码器：GPT的核心模块，输入dec_inputs，输出dec_outputs
  #================构造================
  # 词嵌入层
  # 位置编码层
  # 嵌入层dropout
  # n_layers个解码器层, decoderlayer.py
  # ===================================
class Decoder(nn.Module): 
  def __init__(self,vocab_size, max_seq_len):
    super().__init__()
    # 词嵌入层，输入词表大小，每个词的维度，用于表示词的信息
    self.dec_src_emb = nn.Embedding(vocab_size, d_embedding)
    # 位置编码层，输入最大序列长度，每个位置的维度，用于表示位置信息
    self.dec_pos_emb = nn.Embedding(max_seq_len, d_embedding)
    # 给位置编码层做正态分布初始化，控制初始权重的大小和分布
    nn.init.normal_(self.dec_pos_emb.weight, mean=0.0, std=0.02)
    # 嵌入层dropout,从源头抑制词嵌入层和编码层参数过拟合
    self.emb_dropout = nn.Dropout(p=dropout)
    # 初始化n_layers个解码器层
    self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

  def forward(self, dec_inputs):
      # 创建位置信息, arange创建dec_inputs长度范围内的索引，positions的形状为(1, seq_len)
      positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device).unsqueeze(0)
      # 将位置信息复制到batch_size个样本中，生成的维度是(batch_size, seq_len)
      positions = positions.repeat(dec_inputs.size(0), 1)
      # 词嵌入层和位置编码层的输出相加
      inputs_embedding = self.dec_src_emb(dec_inputs) + self.dec_pos_emb(positions)
      # 嵌入层dropout
      inputs_embedding = self.emb_dropout(inputs_embedding) 
      # 生成自注意力掩码
      attn_mask = self.get_attn_subsequent_mask(inputs_embedding)
      dec_outputs = inputs_embedding
      # 逐层解码
      for layer in self.decoder_layers:
          dec_outputs = layer(dec_outputs, attn_mask)
      return dec_outputs
  
  # 生成自注意力掩码
  def get_attn_subsequent_mask(self, seq):
    # seq形状是 [batch_size, seq_len]
    seq_len = seq.size(1)
    # 使用 torch.triu 生成上三角矩阵
    # diagonal=1 表示保留主对角线以下的部分（不含对角线），上三角置为 1
    subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=seq.device), diagonal=1).bool()
    # 扩展 batch 维度: [seq_len, seq_len] -> [batch_size, seq_len, seq_len]
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(seq.size(0), -1, -1)
    return subsequent_mask

        