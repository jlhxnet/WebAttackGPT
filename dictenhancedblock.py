import torch
import torch.nn as nn
from attentiondictionary import AttentionDictionary

class DictEnhancedBlock(nn.Module):
    """
    独立插件：
    输入特征 → 经过字典 → 得到相似度 → 对特征进行加权增强
    不修改自注意力
    不破坏梯度
    纯残差、纯加性
    """
    def __init__(self, embed_size, dict_size=16):
        super().__init__()
        self.dict = AttentionDictionary(embed_size, dict_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # 得到相似度权重
        weight = self.dict(x)
        
        # 只做增强：相似特征放大，不抑制任何梯度
        enhanced = x * (1.0 + weight)
        
        # 残差连接
        return self.norm(enhanced + x)