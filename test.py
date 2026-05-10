# ===============================================
# test.py
# 推理（覆盖XSS/SQLi/CMDI/正常流量）
# ===============================================

import torch
from common import *
from gpt import GPT

test_payloads = [
    # 🔥 核心XSS样本（空格已补全，特征完整）
    "ID=PAROS%2522%2bSTYLE%253d%2522BACKGROUND%253aURL%2528JAVASCRIPT%253aALERT%2528%2527pAROS%2527%2529%2529%26ID%3d2"
]

# 加载词表与模型
vocab = torch.load(vocab_path, map_location=device)
vocab.set_default_index(vocab["<unk>"])

model = GPT(len(vocab), max_seq_len).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✅ 模型&词表加载成功 | 设备：{device}")
print("="*80)
print(f"🔥 WebAttackGPT 攻击意图研判模型")
print("="*80)
# 批量测试
for payload in test_payloads:
    result = generate_detection(payload, model, vocab)
    print(f"🔹 输入Payload: {payload}")
    print(f"✅ 检测研判: {result}")
    print("-"*80)