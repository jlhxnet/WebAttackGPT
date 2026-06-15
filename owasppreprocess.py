import os
import sys
os.environ["TORCH_USE_NUMPY"] = "0"
os.environ["NUMPY_DISABLE_WARNINGS"] = "1"

import re
import random
from collections import defaultdict

# 导入所有路径
from common import (
    dataset_path,
    owasp_dataset_path,
    MIN_PAYLOAD_LENGTH
)

# ===================== 配置 =====================
FULL_OUTPUT = os.path.join(dataset_path, "owasp_payload.txt")
TARGET_TOTAL = 15000  # 总数量严格控制 15000

# ===================== 分类规则 =====================
def get_label(h_txt):
    if re.search(r'\[id "941\d+', h_txt):
        return "xss", 1
    elif re.search(r'\[id "942\d+', h_txt):
        return "sqli", 1
    elif re.search(r'\[id "93[0123]\d+', h_txt):
        return "cmdi", 1
    else:
        return "normal", 0

# ===================== 提取Payload =====================
def extract_payload(raw_b):
    lines = raw_b.strip().splitlines()
    if not lines:
        return None
    first_line = lines[0].strip()
    if first_line.startswith("GET") or first_line.startswith("POST"):
        parts = first_line.split()
        if len(parts) >= 2:
            return parts[1]
    return None

# ===================== 全量解析 =====================
def parse_one_log(path, fw, counter):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r'--[0-9a-fA-F]+-A--', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        b_match = re.search(r'--.*?-B--(.*?)(?=--|$)', block, re.DOTALL)
        h_match = re.search(r'--.*?-H--(.*?)(?=--|$)', block, re.DOTALL)
        if not b_match:
            continue

        payload = extract_payload(b_match.group(1))
        if not payload or len(payload) < MIN_PAYLOAD_LENGTH:
            continue

        h_content = h_match.group(1) if h_match else ""
        typ, label = get_label(h_content)
        fw.write(f"{payload},{typ},{label}\n")
        counter[typ] += 1

    print(f"✅ {os.path.basename(path)}")

# ===================== ✅ xss+sqli全选，剩余 cmdi70% / normal30% =====================
def build_balanced_15k():
    with open(FULL_OUTPUT, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    cls_dict = defaultdict(list)
    for line in lines:
        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            continue
        payload, typ, label = parts
        cls_dict[typ].append(line)

    # 取出各类
    xss = cls_dict["xss"]
    sqli = cls_dict["sqli"]
    cmdi = cls_dict["cmdi"]
    normal = cls_dict["normal"]

    print("\n📊 原始数量：")
    print(f" xss    : {len(xss)} (全选)")
    print(f" sqli   : {len(sqli)} (全选)")
    print(f" cmdi   : {len(cmdi)}")
    print(f" normal : {len(normal)}")

    # 固定：xss + sqli 全选
    selected = xss + sqli
    current = len(selected)
    need = TARGET_TOTAL - current

    # 剩余部分：cmdi 70%，normal 30%
    need_cmdi = int(need * 0.7)
    need_normal = need - need_cmdi

    # 随机采样
    sel_cmdi = random.sample(cmdi, min(need_cmdi, len(cmdi)))
    sel_normal = random.sample(normal, min(need_normal, len(normal)))

    selected += sel_cmdi + sel_normal
    random.shuffle(selected)

    print("\n✅ 最终 15000 条配比完成：")
    print(f" xss    : {len(xss)}")
    print(f" sqli   : {len(sqli)}")
    print(f" cmdi   : {len(sel_cmdi)}")
    print(f" normal : {len(sel_normal)}")
    print(f" 总计   : {len(selected)}")

    return selected

# ===================== 7:1.5:1.5 拆分 =====================
def split_train_test(data):
    random.seed(42)
    random.shuffle(data)
    n = len(data)

    train = data[:int(n*0.7)]
    valid = data[int(n*0.7):int(n*0.85)]
    test  = data[int(n*0.85):]

    def write(name, lst):
        path = os.path.join(dataset_path, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lst))
        print(f"✅ {name} → {len(lst)} 条")

    write("owasp_train.txt", train)
    write("owasp_valid.txt", valid)
    write("owasp_test.txt", test)

# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(dataset_path, exist_ok=True)
    counter = defaultdict(int)

    print("🔵 全量采集 OWASP 数据...")
    with open(FULL_OUTPUT, "w", encoding="utf-8") as fw:
        for rt, _, files in os.walk(owasp_dataset_path):
            for fname in files:
                if fname == "modsec_audit.anon.log":
                    parse_one_log(os.path.join(rt, fname), fw, counter)

    print(f"\n🎉 全量采集完成！")
    final_data = build_balanced_15k()

    print("\n🔪 开始 7:1.5:1.5 拆分...")
    split_train_test(final_data)

    print("\n🎉 全部完成！")