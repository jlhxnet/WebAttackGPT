# ==========================================================================
# AttentionDictionary.py
# 用于Web攻击载荷检测的轻量级词典增强注意力模块。
# 构建可学习的特征原型字典，聚类正常流量与攻击流量模式。
# 强化高风险恶意符号的注意力权重，抑制无关噪声干扰。
# 提升长尾攻击（如命令注入）的检测召回率，同时保持模型轻量化。
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata

class AttentionDictionary(nn.Module):
    def __init__(self, embed_size, dict_size=32, boost_factor=1.2):
        super().__init__()
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.boost_factor = boost_factor

        # ==========================================================================
        # 全局统一高危符号：覆盖 CMDI + XSS + SQLi 所有核心攻击符号
        # 只要出现这些符号 → 整个上下文【禁止减权】，保证攻击检出
        # ==========================================================================
        self.GLOBAL_ATTACK_TOKENS = set([
            # --- CMDI 核心 ---
            ord(';'), ord('|'), ord('&'), ord('$'), ord('`'), ord('\\'),
            ord('('), ord(')'), ord('/'), ord('?'),
            
            # --- XSS 核心 ---
            ord('<'), ord('>'), ord('='), ord('"'), ord('\''),
            ord('['), ord(']'), ord('{'), ord('}'),
            
            # --- SQLi 核心 ---
            ord('\''), ord('"'), ord('#'), ord('*'), ord('=')
        ])

        # ---------------- 超参数 ----------------
        self.lang_char_scale = 0.15     # 自然语言字母（全语种）减权系数
        self.mild_punct_scale = 0.3      # 普通标点/URL符号减权系数
        self.ctx_window = 2              # 攻击上下文窗口 ±2 字符

        self.prototypes = nn.Parameter(torch.randn(dict_size, embed_size))
        self.proj = nn.Linear(embed_size, embed_size)
        self.contrast_proj = nn.Linear(embed_size, embed_size)
        nn.init.normal_(self.prototypes, std=0.02)
        
        self.gate = nn.Sequential(
            nn.Linear(embed_size, embed_size // 4),
            nn.SiLU(),
            nn.Linear(embed_size // 4, 1),
            nn.Sigmoid()
        )

    # ==========================================================================
    # 全语种自动识别：英语/西语/德语/阿语/俄语/日语... 
    # ==========================================================================
    def is_natural_lang_char(self, token_id):
        try:
            cat = unicodedata.category(chr(token_id))
            return cat.startswith(('L', 'M'))  # L=字母, M=重音/变音
        except:
            return False

    # ==========================================================================
    # 普通无害标点（URL、表单、密码常用符号）
    # ==========================================================================
    def is_mild_punct(self, token_id):
        return token_id in {
            ord('_'), ord('-'), ord('.'), ord(','), ord(':'),
            ord('%'), ord('='), ord('#'), ord('@')
        }

    # ==========================================================================
    # 全局攻击上下文：只要附近有 CMDI/XSS/SQLi 符号 → 禁止减权
    # ==========================================================================
    def get_attack_context_mask(self, token_ids):
        B, T = token_ids.shape
        device = token_ids.device
        
        # 标记所有攻击符号位置
        attack_mask = torch.zeros_like(token_ids, dtype=torch.float32, device=device)
        for t in self.GLOBAL_ATTACK_TOKENS:
            attack_mask = torch.logical_or(attack_mask, (token_ids == t))
        
        # 滑动窗口扩散上下文（左右 ±2）
        ctx_mask = attack_mask.clone()
        for d in range(1, self.ctx_window + 1):
            if d >= T: continue
            ctx_mask[:, :-d] = torch.max(ctx_mask[:, :-d], attack_mask[:, d:])
            ctx_mask[:, d:] = torch.max(ctx_mask[:, d:], attack_mask[:, :-d])
        
        return ctx_mask.unsqueeze(-1)  # [B, T, 1]

    # ==========================================================================
    # 前向推理（核心逻辑）
    # ==========================================================================
    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        x_proj = self.proj(x)
        x_norm = F.normalize(x_proj, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(x_norm, proto_norm.T)
        sim_max = sim.max(dim=-1, keepdim=True)[0]

        cfeat = self.contrast_proj(x)
        cnorm = F.normalize(cfeat, dim=-1)
        csim = torch.matmul(cnorm, proto_norm.T)
        csim_max = csim.max(dim=-1, keepdim=True)[0]
        fused = (sim_max + csim_max) / 2.0
        
        g = self.gate(x)
        weight = fused * g
        if token_ids is not None:
            device = token_ids.device
            # 1. 攻击符号加权
            attack_mask = torch.zeros_like(token_ids, dtype=torch.float32, device=device)
            for t in self.GLOBAL_ATTACK_TOKENS:
                attack_mask = torch.logical_or(attack_mask, (token_ids == t))
            attack_mask = attack_mask.unsqueeze(-1)
            weight = weight * (1.0 + (self.boost_factor - 1.0) * attack_mask)

            # 2. 攻击上下文掩码：在上下文内 = 不减权
            attack_ctx_mask = self.get_attack_context_mask(token_ids)
            safe_zone = 1.0 - attack_ctx_mask  # 无攻击区域 = 可以减权

            # 3. 全语种自然语言字符：仅安全区减权
            lang_mask = torch.tensor(
                [[self.is_natural_lang_char(t.item()) for t in seq] for seq in token_ids],
                dtype=torch.float32, device=device
            ).unsqueeze(-1) * safe_zone
            weight = weight * (1.0 - (1.0 - self.lang_char_scale) * lang_mask)

            # 4. 普通标点/URL符号：仅安全区减权
            punct_mask = torch.tensor(
                [[self.is_mild_punct(t.item()) for t in seq] for seq in token_ids],
                dtype=torch.float32, device=device
            ).unsqueeze(-1) * safe_zone
            weight = weight * (1.0 - (1.0 - self.mild_punct_scale) * punct_mask)

        return torch.sigmoid(weight)