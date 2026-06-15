# ====================================================
# evaluate_AARE.py
# 对抗样本鲁棒性评估脚本
# 扰动规则：不可见字符插入 / 多层URL编码 / 关键字拆分 / 冗余噪声注入
# =====================================================
import torch
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import time
import random
import urllib.parse
from common import *
from gpt import GPT
from dataset import AttackDataset

# ===================== 全局配置 =====================
LABEL_NAMES = ["normal", "xss", "sqli", "cmdi"]

# 对抗扰动相关配置
INVISIBLE_CHARS = ["\t", "\r", "\u200B"]
ATTACK_KEYWORDS = {
    "normal": [],
    "xss": ["script", "alert", "onclick", "onerror", "<", ">"],
    "sqli": ["select", "union", "and", "or", "'", "\"", "--"],
    "cmdi": ["ping", "ls", "whoami", ";", "|", "&"]
}
# 固定随机种子，保证对抗样本可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================== 对抗扰动函数 =====================
def add_invisible_char(payload: str) -> str:
    """扰动1：随机插入不可见字符"""
    char_list = list(payload)
    insert_cnt = random.randint(1, 3)
    for _ in range(insert_cnt):
        pos = random.randint(0, len(char_list) - 1)
        char_list.insert(pos, random.choice(INVISIBLE_CHARS))
    return "".join(char_list)

def multi_url_encode(payload: str) -> str:
    """扰动2：1~2层URL编码"""
    encode_times = random.randint(1, 2)
    text = payload
    for _ in range(encode_times):
        text = urllib.parse.quote(text)
    return text

def split_keyword(payload: str, label: str) -> str:
    """扰动3：攻击关键字拆分，插入无害标点"""
    kw_list = ATTACK_KEYWORDS.get(label.lower(), [])
    split_sym = [",", ".", "_", "-"]
    text = payload
    for kw in kw_list:
        if kw in text and len(kw) > 1:
            split_pos = random.randint(1, len(kw) - 1)
            new_kw = kw[:split_pos] + random.choice(split_sym) + kw[split_pos:]
            text = text.replace(kw, new_kw, 1)
    return text

def add_redundant_noise(payload: str) -> str:
    """扰动4：首尾拼接冗余URL参数"""
    noise_pool = ["id=1", "page=2", "sort=asc", "limit=10", "type=normal"]
    pre_noise = "&".join(random.sample(noise_pool, random.randint(1, 2)))
    suf_noise = "&".join(random.sample(noise_pool, random.randint(1, 2)))
    return f"{pre_noise}&{payload}&{suf_noise}"

def create_adversarial_sample(payload: str, label: str) -> str:
    """单条样本生成对抗载荷，随机组合1~2种扰动"""
    func_pool = [add_invisible_char, multi_url_encode, split_keyword, add_redundant_noise]
    select_funcs = random.sample(func_pool, k=random.randint(1, 2))
    adv_text = payload
    for func in select_funcs:
        if func.__name__ == "split_keyword":
            adv_text = func(adv_text, label)
        else:
            adv_text = func(adv_text)
    return adv_text

# ===================== 加载原始测试集 =====================
def get_test_data(vocab):
    test_dataset = AttackDataset(vocab, mode="test")
    test_payloads = test_dataset.raw_samples
    test_true_labels = test_dataset.augmented_type_list
    return test_payloads, test_true_labels

# ===================== 从生成结果提取标签 =====================
def get_pred_label(pred_text):
    if "xss攻击" in pred_text:
        return "xss"
    elif "sqli攻击" in pred_text:
        return "sqli"
    elif "cmdi攻击" in pred_text:
        return "cmdi"
    else:
        return "normal"

# ===================== 主评估流程 =====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 WebAttackGPT 对抗样本鲁棒性评估")
    print("=" * 80)

    # 加载词表
    vocab = torch.load(vocab_path, map_location=device)
    print(f"✅ 词表加载完成 | 词表大小：{len(vocab)}")

    # 加载模型
    model = GPT(len(vocab), max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载完成 | 设备：{device}")

    # 加载原始测试集
    raw_payloads, raw_true_labels = get_test_data(vocab)
    print(f"✅ 原始测试集加载完成 | 总样本数：{len(raw_payloads)}")

    # 生成对抗样本
    print(f"\n🔨 开始生成对抗载荷...")
    adv_payloads = []
    adv_true_labels = []
    for payload, label in zip(raw_payloads, raw_true_labels):
        adv_p = create_adversarial_sample(payload, label)
        adv_payloads.append(adv_p)
        adv_true_labels.append(label)
    print(f"✅ 对抗样本生成完成 | 总样本数：{len(adv_payloads)}")

    # 混合数据集：原始样本 + 对抗样本
    all_payloads = raw_payloads + adv_payloads
    all_true_labels = raw_true_labels + adv_true_labels
    total_num = len(all_payloads)
    print(f"✅ 混合对抗测试集加载完成 | 总样本数：{total_num}")

    # 批量推理
    print(f"\n🔍 开始批量推理...")
    test_pred_labels = []
    start_time = time.time()

    for idx, payload in enumerate(all_payloads):
        pred_text = generate_detection(payload, model, vocab)
        pred_label = get_pred_label(pred_text)
        test_pred_labels.append(pred_label)
        if (idx + 1) % 50 == 0:
            print(f"   已推理：{idx + 1}/{total_num}")

    # 统计推理耗时与速度
    end_time = time.time()
    total_infer_time = end_time - start_time
    infer_speed = total_num / total_infer_time

    print(f"\n⏱️  推理速度统计：")
    print(f"   总样本数：{total_num} 个")
    print(f"   总推理耗时：{total_infer_time:.2f} 秒")
    print(f"   🚀 推理速度：{infer_speed:.2f} 样本/秒")

    # 计算评估指标
    print(f"\n" + "=" * 80)
    print(f"📊 对抗样本测试集指标报告")
    print("=" * 80)

    accuracy = accuracy_score(all_true_labels, test_pred_labels)
    cls_report = classification_report(
        all_true_labels, test_pred_labels,
        labels=LABEL_NAMES,
        target_names=LABEL_NAMES,
        digits=4, output_dict=True
    )
    cls_report_print = classification_report(
        all_true_labels, test_pred_labels,
        labels=LABEL_NAMES,
        target_names=LABEL_NAMES,
        digits=4
    )
    print(f"🎯 整体准确率：{accuracy:.4f}")
    print(f"\n📋 各类攻击详细指标：")
    print(cls_report_print)

    # 仅保存对抗评估指标（单一文件）
    with open("adv_eval_result.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WebAttackGPT 对抗样本评估结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"整体准确率: {accuracy:.4f}\n\n")
        f.write(cls_report_print)
        f.write(f"\n总测试样本数: {total_num}\n")
        f.write(f"总推理耗时: {total_infer_time:.2f}s\n")
        f.write(f"推理速度: {infer_speed:.2f} 样本/秒\n")
    print(f"✅ 对抗评估指标已保存到: adv_eval_result.txt")

    print("=" * 80)
    print("🎉 对抗样本评估全部完成！")
    print("=" * 80)