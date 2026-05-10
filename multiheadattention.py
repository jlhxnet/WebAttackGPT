# ==================================================================
# multiheadattention.py
# 多头自注意力层：输入Q,K,V,attn_mask，输出注意力计算后的output,weights
# Q线性映射层
# K线性映射层
# V线性映射层
# 缩放点积注意力层，定义scaleddotproductattention.py
# 线性映射层
# ===================================================================

from scaleddotproductattention import *
from common import *

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Q线性映射层，输入维度是d_embedding，输出维度是n_heads*d_k
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        # K线性映射层，输入维度是d_embedding，输出维度是n_heads*d_k
        self.W_K = nn.Linear(d_embedding, d_k * n_heads) 
        # V线性映射层，输入维度是d_embedding，输出维度是n_heads*d_v
        self.W_V = nn.Linear(d_embedding, d_v * n_heads) 
        # 线性映射层，输入维度是n_heads*d_v，输出维度是d_embedding
        self.linear = nn.Linear(d_v * n_heads, d_embedding)
        # 缩放点积注意力层
        self.scaleddotproductattention = ScaledDotProductAttention()
        # 权重初始化（让模型更快收敛）
        nn.init.normal_(self.W_Q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_K.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_V.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self,Q,K,V,attn_mask):
        batch_size = Q.size(0)
        # (batch_size, seq_len, d_embedding) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q_s = self.W_Q(Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size,-1,n_heads,d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size,-1,n_heads,d_v).transpose(1,2)
        # mask维度扩展,(batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1) 
        # 缩放点积注意力, 结果是(batch_size, n_heads, seq_len, d_v)
        context, weights = self.scaleddotproductattention(q_s,k_s,v_s,attn_mask)
        # 拼接多头结果, (batch_size, n_heads, seq_len, d_v) -> (batch_size, seq_len, n_heads, d_v) -> (batch_size, seq_len, n_heads*d_v)
        context = context.transpose(1,2).contiguous().view(batch_size,-1,n_heads*d_v)
        # 线性映射还原为d_embedding维度，输出是(batch_size, seq_len, d_embedding)
        output = self.linear(context)
        return output, weights

