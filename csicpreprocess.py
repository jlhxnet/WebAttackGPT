# ===================================
# csicpreprocess.py
# 负责原始数据文件拆分为csv文件
# ===================================

import re
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from common import dataset_path
import random

# 原始文件路径
NORMAL_RAW = os.path.join(dataset_path, "normalTrafficTraining.txt")
ATTACK_RAW = os.path.join(dataset_path, "anomalousTrafficTest.txt")

# ===================== 工具函数 =====================
# 加载数据列表
def load_reqs(file):
    with open(file, "r", encoding="latin-1") as f:
      # 返回以行分隔的列表
      return [r.strip() for r in f.read().split("\n\n") if r.strip()]

# 提取payload
def get_payload(req):
  # 正则表达式：匹配GET请求里？后面的参数
  get = re.search(r"GET\s+[^?]*\?([^\s]+)\s+HTTP", req)
  if get: return get.group(1).strip()
  # 正则表达式：匹配POST请求里？后面的参数
  post = re.search(r"POST.*?\n\n([^\n]+)", req, re.DOTALL)
  return post.group(1).strip() if post else ""

# 提取攻击类型
def get_attack_type(type):
  if not type: return "normal"
  if re.search(r"('|\";|--|UNION|SELECT)", type, re.I): return "sqli"
  if re.search(r"(<script|alert|onerror)", type, re.I): return "xss"
  if re.search(r"(\.\./|etc/passwd)", type, re.I): return "path_traversal"
  if re.search(r"(ping|ls|;|&|\|)", type, re.I): return "cmdi"
  return "normal"

# 轻量数据增强（攻击样本）
def augment_attack_payload(payload):
    variants = set()
    variants.add(payload)
    # 变体1：大小写转换
    variants.add(payload.swapcase())
    # 变体2：空格换成URL编码%20
    variants.add(payload.replace(" ", "%20"))
    # 变体3：末尾加无害字符
    variants.add(payload + "123")
    return list(variants)

# 轻量数据增强（正常样本）
def augment_normal_payload(payload):
    variants = set()
    variants.add(payload)  # 保留原始

    # 3种温和的正常流量增强
    aug_list = [
        lambda s: s.lower(),                  # 小写
        lambda s: s.replace(" ", "_"),        # 空格变下划线
        lambda s: s + "123"                   # 加数字后缀
    ]

    # 随机选 1 种 → 总数≈1.5倍
    chosen = random.sample(aug_list, 1)
    for func in chosen:
        variants.add(func(payload))

    return list(variants)

# 数据增强汇总
def augment_class(df_class, is_normal=False):
    augmented = []
    for _, row in df_class.iterrows():
        # normal 用专门的随机增强
        if is_normal:
            vars_list = augment_normal_payload(row["payload"])
        # 攻击用原来的增强
        else:
            vars_list = augment_attack_payload(row["payload"])
        
        for var_payload in vars_list:
            augmented.append({
                "payload": var_payload,
                "attack_type": row["attack_type"],
                "label": row["label"]
            })
    return pd.DataFrame(augmented)

# ===================== 构建原始数据集 =====================
print("🔨 加载原始数据...")
normal_data = []
attack_data = []

# 从文件中加载正常流量，生成格式：payload，attack_type,label
for req in load_reqs(NORMAL_RAW):
    p = get_payload(req)
    if p: normal_data.append({"payload": p, "attack_type": "normal", "label": 0})

# 从文件中加载攻击流量，生成格式：payload，attack_type,label
for req in load_reqs(ATTACK_RAW):
    p = get_payload(req)
    if p:
        atk_type = get_attack_type(p)
        attack_data.append({"payload": p, "attack_type": atk_type, "label": 1})
# 合并数据集，将列表转换为DataFrame，并添加列名
df = pd.DataFrame(normal_data + attack_data)
print("\n📊 原始分布：")
# value_counts() 统计每个类别的数量
print(df["attack_type"].value_counts())

# ===================== 温和均衡 + 少样本增强 =====================
# 将df中attack_type列的值为normal的行提取出来，并赋值给normal变量
normal = df[df["attack_type"] == "normal"]
sqli = df[df["attack_type"] == "sqli"]
cmdi = df[df["attack_type"] == "cmdi"]
xss = df[df["attack_type"] == "xss"]

# normal：温和随机增强（1.5倍）
normal_aug = augment_class(normal, is_normal=True)
# 攻击：保持原来的增强
xss_aug = augment_class(xss)
sqli_aug = augment_class(sqli)
cmdi_final = cmdi

# 合并攻击集（真实+增强变体）
attack_final = pd.concat([sqli_aug, cmdi_final, xss_aug])

# 最终数据集
final_df = pd.concat([normal_aug, attack_final]).sample(frac=1, random_state=42)

print("\n✅ 增强+均衡后分布：")
print(final_df["attack_type"].value_counts())

# ===================== 拆分训练/验证/测试 =====================
# train_test_split返回train, test两个DataFrame
train, temp = train_test_split(final_df, test_size=0.3, stratify=final_df["attack_type"], random_state=42)
# train_test_split返回valid, test两个DataFrame
valid, test = train_test_split(temp, test_size=0.5, stratify=temp["attack_type"], random_state=42)

# 保存
train.to_csv(os.path.join(dataset_path, "train.csv"), index=False)
valid.to_csv(os.path.join(dataset_path, "valid.csv"), index=False)
test.to_csv(os.path.join(dataset_path, "test.csv"), index=False)

print(f"\n🎉 生成成功！路径：{dataset_path}")
print(f"训练集: {len(train)} | 验证集: {len(valid)} | 测试集: {len(test)}")