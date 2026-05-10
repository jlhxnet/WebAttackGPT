import torch
import torch.optim as optim
from gpt import GPT
from torch.utils.data import DataLoader, WeightedRandomSampler
from csicdataset import AttackDataset
import torch.nn as nn
from common import *
import os
import matplotlib.pyplot as plt
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

# 构建词表
vocab = build_vocab_from_iterator(AttackDataset.yield_attack_tokens(), specials=SPECIAL_TOKENS)
# 设置默认索引为unk
vocab.set_default_index(vocab["<unk>"])
torch.save(vocab, vocab_path)
print(f"✅ 攻击日志词表构建完成 | 大小：{len(vocab)}")

# 数据加载
train_dataset = AttackDataset(vocab, mode="train")
valid_dataset = AttackDataset(vocab, mode="valid")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_fn(x, vocab))
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x: collate_fn(x, vocab))

# 模型初始化
model = GPT(len(vocab), max_seq_len).to(device)
print(f"✅ 模型初始化完成 | 总参数：{sum(p.numel() for p in model.parameters())}")

# 优化器初始化
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)
# 余弦退火学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ====================================== 训练循环 =============================================
train_losses = []
valid_losses = []
# 最佳验证损失
best_val_loss = float('inf')
# 交互式绘图
plt.ion()

for epoch in range(epochs):
    # 训练：标准GPT全序列生成损失
    model.train()
    total_train_loss = 0
    for batch_idx, (src, tgt) in enumerate(train_dataloader):
      src, tgt = src.to(device), tgt.to(device)
      optimizer.zero_grad()
      outputs = model(src)

      # 标准生成式损失：全序列计算，忽略padding
      loss = F.cross_entropy(
          outputs.reshape(-1, outputs.size(-1)),  # 展平序列
          tgt.reshape(-1),                        # 展平目标
          ignore_index=vocab["<pad>"]             # 忽略填充符
      )

      loss.backward()
      optimizer.step()
      total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    scheduler.step()
    train_losses.append(avg_train_loss)

    # 验证：标准GPT全序列损失
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for src, tgt in valid_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src)

            # 验证损失和训练完全一致，全序列计算
            val_loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                tgt.reshape(-1),
                ignore_index=vocab["<pad>"]
            )
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(valid_dataloader)
    valid_losses.append(avg_val_loss)

    # 保存 & 早停
    print(f"Epoch {epoch+1}/{epochs} | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss and avg_val_loss > 0:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_path)
        print(f"💾 最佳模型保存")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"⚠️ 早停计数：{early_stop_counter}/{early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print("🚨 早停触发")
            break

    # 损失曲线绘制
    plt.figure(figsize=(14,7))
    plt.plot(train_losses, label='train', marker='o')
    plt.plot(valid_losses, label='valid', marker='s')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_curve_path)
    plt.close()