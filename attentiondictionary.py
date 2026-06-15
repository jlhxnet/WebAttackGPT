import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata

class AttentionDictionary(nn.Module):
    def __init__(self, embed_size, dict_size=48, boost_factor=1.2):
        super().__init__()
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.boost_factor = boost_factor

        self.GLOBAL_ATTACK_TOKENS = set([
            ord(';'), ord('|'), ord('&'), ord('$'), ord('`'), ord('\\'),
            ord('('), ord(')'), ord('/'), ord('?'),
            ord('<'), ord('>'), ord('='), ord('"'), ord('\''),
            ord('['), ord(']'), ord('{'), ord('}'),
            ord('#'), ord('*')
        ])

        self.lang_char_scale = 0.15
        self.mild_punct_scale = 0.3
        self.ctx_window = 4

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

    def is_natural_lang_char(self, token_id):
        try:
            cat = unicodedata.category(chr(token_id))
            return cat.startswith(('L', 'M'))
        except:
            return False

    def is_mild_punct(self, token_id):
        return token_id in {
            ord('_'), ord('-'), ord('.'), ord(','), ord(':'),
            ord('%'), ord('#'), ord('@')
        }

    def get_attack_context_mask(self, token_ids):
        B, T = token_ids.shape
        device = token_ids.device
        
        attack_mask = torch.zeros_like(token_ids, dtype=torch.float32, device=device)
        for t in self.GLOBAL_ATTACK_TOKENS:
            attack_mask = torch.logical_or(attack_mask, (token_ids == t))
        
        ctx_mask = attack_mask.clone()
        for d in range(1, self.ctx_window + 1):
            if d >= T: continue
            ctx_mask[:, :-d] = torch.max(ctx_mask[:, :-d], attack_mask[:, d:])
            ctx_mask[:, d:] = torch.max(ctx_mask[:, d:], attack_mask[:, :-d])
        
        return ctx_mask.unsqueeze(-1)

    def forward(self, x, token_ids=None):
        B, T, C = x.shape

        x_proj = self.proj(x)
        win_size = 3
        pad = 1
        x_pad = F.pad(x_proj, [0,0,pad,pad])
        
        # ===================== MAX 池化 =====================
        x_ngram = F.max_pool1d(
            x_pad.transpose(1, 2), 
            kernel_size=win_size, 
            stride=1
        ).transpose(1, 2)
        # =====================================================================

        x_norm = F.normalize(x_ngram, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(x_norm, proto_norm.T)
        sim_max = sim.max(dim=-1, keepdim=True)[0]

        cfeat = self.contrast_proj(x)
        c_pad = F.pad(cfeat, [0,0,pad,pad])
        
        # ===================== MAX 池化 =====================
        c_ngram = F.max_pool1d(
            c_pad.transpose(1, 2), 
            kernel_size=win_size, 
            stride=1
        ).transpose(1, 2)
        # =====================================================================
        
        cnorm = F.normalize(c_ngram, dim=-1)
        csim = torch.matmul(cnorm, proto_norm.T)
        csim_max = csim.max(dim=-1, keepdim=True)[0]

        fused = (sim_max + csim_max) / 2.0
        g = self.gate(x)
        weight = fused * g

        if token_ids is not None:
            device = token_ids.device
            attack_mask = torch.zeros_like(token_ids, dtype=torch.float32, device=device)
            for t in self.GLOBAL_ATTACK_TOKENS:
                attack_mask = torch.logical_or(attack_mask, (token_ids == t))
            attack_mask = attack_mask.unsqueeze(-1)
            weight = weight * (1.0 + (self.boost_factor - 1.0) * attack_mask)

            attack_ctx_mask = self.get_attack_context_mask(token_ids)
            safe_zone = 1.0 - attack_ctx_mask

            lang_mask = torch.tensor(
                [[self.is_natural_lang_char(t.item()) for t in seq] for seq in token_ids],
                dtype=torch.float32, device=device
            ).unsqueeze(-1) * safe_zone
            weight = weight * (1.0 - (1.0 - self.lang_char_scale) * lang_mask)

            punct_mask = torch.tensor(
                [[self.is_mild_punct(t.item()) for t in seq] for seq in token_ids],
                dtype=torch.float32, device=device
            ).unsqueeze(-1) * safe_zone
            weight = weight * (1.0 - (1.0 - self.mild_punct_scale) * punct_mask)

        return torch.sigmoid(weight)