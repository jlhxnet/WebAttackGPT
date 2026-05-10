#==========================================================================
# decoderlayer.py
# 解码器层：构成Decoder， 输入(dec_inputs, attn_mask)，输出dec_outputs
# 多头自注意力层，multiheadattention.py
# 前馈网络层, poswisefeedforwardnet.py
# 归一化层
# 层间Dropout（防止梯度爆炸）
#============================================================================

from multiheadattention import MultiHeadAttention
import torch.nn as nn
from poswisefeedforwardnet import PositionwiseFeedForwardNet
from common import *
from dictenhancedblock import DictEnhancedBlock

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 多头自注意力层
        self.self_attn = MultiHeadAttention()
        # 前馈网络层
        self.feed_forward = PositionwiseFeedForwardNet(d_embedding, hidden_size, dropout)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_embedding)
        self.norm2 = nn.LayerNorm(d_embedding)
        # 层间Dropout（防止梯度爆炸）
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        # 词典增强层
        self.dict_enhance = DictEnhancedBlock(d_embedding)
        
    def forward(self, dec_inputs, attn_mask = None):
      # 保存原始输入作为残差
      residual = dec_inputs  
      # 先归一化，再做注意力
      x = self.norm1(residual)
      # 多头自注意力层，输入dec_inputs, dec_inputs, dec_inputs, attn_mask，输出经过注意力计算后的attn_output, attn_weights
      attn_output, _ = self.self_attn(x, x, x, attn_mask)
      # 残差+dropout
      x = residual + self.dropout1(attn_output)
      x = self.dict_enhance(x)
      # 前馈网络，同上
      residual = x
      x = self.norm2(residual)
      ff_outputs = self.feed_forward(x)
      x = residual + self.dropout2(ff_outputs)

      return x