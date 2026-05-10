# ==============================================
# gpt.py
# LiteGPT
# 解码器，decoder.py
# 全连接层
# ==============================================

import torch.nn as nn
from decoder import Decoder
from common import *

class GPT(nn.Module):
  def __init__(self, vocab_size, max_seq_len):
    super().__init__()
    # 构建解码器, 输入词表大小和最大序列长度
    self.decoder = Decoder(vocab_size, max_seq_len)
    # 全连接层，输入维度是d_embedding，输出维度是词表大小，代表预测结果
    self.projection = nn.Linear(d_embedding, vocab_size)

  def forward(self, dec_inputs):
    # 将输入数据传递给解码器
    dec_outputs = self.decoder(dec_inputs)
    # 将解码器输出传递给全连接层生成预测结果,dec_logits的形状为(batch_size, seq_len, vocab_size)
    dec_logits = self.projection(dec_outputs)
    return dec_logits