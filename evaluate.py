# ====================================================
# evaluate.py
# 评估脚本：用于评估WebAttackGPT模型在CSIC数据集上的性能
# =====================================================

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
from common import *
from gpt import GPT
from csicdataset import AttackDataset

# ===================== 标签映射 =====================
LABEL_NAMES = ["normal", "xss", "sqli", "cmdi"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 图配色

# ===================== 加载测试集 =====================
def get_test_data(vocab):
    test_dataset = AttackDataset(vocab, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    test_payloads = []
    test_true_labels = []
    for _, row in test_dataset.df.iterrows():
        try:
            payload = str(row["payload"]).strip()
            true_label = str(row["attack_type"]).strip().lower()
            test_payloads.append(payload)
            test_true_labels.append(true_label)
        except Exception:
            continue
    return test_payloads, test_true_labels, test_loader

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

# ===================== 📊 绘制四类指标柱状图 =====================
def plot_metric_bar_chart(report_dict):
    categories = LABEL_NAMES
    precision = [report_dict[c]['precision'] for c in categories]
    recall = [report_dict[c]['recall'] for c in categories]
    f1 = [report_dict[c]['f1-score'] for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label='Precision', color=COLORS[0])
    plt.bar(x, recall, width, label='Recall', color=COLORS[1])
    plt.bar(x + width, f1, width, label='F1-Score', color=COLORS[2])

    plt.xlabel('Attack Types', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Detection Performance of WebAttackGPT on CSIC Dataset', fontsize=14)
    plt.xticks(x, categories)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(metric_bar_chart_path, dpi=300)
    plt.close()
    print("✅ 柱状图已保存：metric_bar_chart.png")

# ===================== 📊 绘制混淆矩阵 =====================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(LABEL_NAMES))
    plt.xticks(tick_marks, LABEL_NAMES, rotation=45)
    plt.yticks(tick_marks, LABEL_NAMES)

    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i,j] < 500 else "white")

    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()
    print("✅ 混淆矩阵已保存：confusion_matrix.png")

# ===================== 主评估流程 =====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 WebAttackGPT 模型 - 测试集全量评估")
    print("=" * 80)

    vocab = torch.load(vocab_path, map_location=device)
    print(f"✅ 词表加载完成 | 词表大小：{len(vocab)}")

    model = GPT(len(vocab), max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载完成 | 设备：{device}")

    test_payloads, test_true_labels, test_loader = get_test_data(vocab)
    print(f"✅ 测试集加载完成 | 总样本数：{len(test_payloads)}")

    print(f"\n🔍 开始批量推理...")
    test_pred_labels = []
    test_pred_texts = []

    # 【新增】推理开始计时
    start_time = time.time()

    for idx, payload in enumerate(test_payloads):
        pred_text = generate_detection(payload, model, vocab)
        pred_label = get_pred_label(pred_text)
        test_pred_labels.append(pred_label)
        test_pred_texts.append(pred_text)
        if (idx + 1) % 50 == 0:
            print(f"   已推理：{idx + 1}/{len(test_payloads)}")

    # 【新增】推理结束计时 + 计算速度
    end_time = time.time()
    total_infer_time = end_time - start_time  # 总推理耗时
    infer_speed = len(test_payloads) / total_infer_time  # 推理速度（样本/秒）

    # 【新增】打印速度结果
    print(f"\n⏱️  推理速度统计：")
    print(f"   总样本数：{len(test_payloads)} 个")
    print(f"   总推理耗时：{total_infer_time:.2f} 秒")
    print(f"   🚀 推理速度：{infer_speed:.2f} 样本/秒")

    print(f"\n" + "=" * 80)
    print(f"📊 模型测试集指标报告")
    print("=" * 80)

    accuracy = accuracy_score(test_true_labels, test_pred_labels)
    cls_report = classification_report(
        test_true_labels, test_pred_labels,
        labels=LABEL_NAMES,
        target_names=LABEL_NAMES,
        digits=4, output_dict=True
    )
    cls_report_print = classification_report(
        test_true_labels, test_pred_labels,
        labels=LABEL_NAMES,
        target_names=LABEL_NAMES,
        digits=4
    )
    print(f"🎯 整体准确率：{accuracy:.4f}")
    print(f"\n📋 各类攻击详细指标：")
    print(cls_report_print)

    # ===================== ✅保存评估指标到文件=====================
    with open(eval_result_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WebAttackGPT 模型评估结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"整体准确率: {accuracy:.4f}\n\n")
        f.write(cls_report_print)
        f.write(f"\n总测试样本数: {len(test_payloads)}\n")
        f.write(f"总推理耗时: {total_infer_time:.2f}s\n")
        f.write(f"推理速度: {infer_speed:.2f} 样本/秒\n")
    print(f"✅ 评估指标已保存到: {eval_result_path}")

    # ===================== ✅保存评估错误样本=====================
    error_count = 0
    with open(eval_error_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("模型预测错误样本记录\n")
        f.write("格式: 请求内容 | 正确分类 | 预测分类 | 模型输出\n")
        f.write("=" * 80 + "\n\n")
        
        for payload, true_label, pred_label, pred_text in zip(
            test_payloads, test_true_labels, test_pred_labels, test_pred_texts
        ):
            if true_label != pred_label:
                error_count += 1
                f.write(f"请求: {payload}\n")
                f.write(f"正确分类: {true_label} | 预测分类: {pred_label}\n")
                f.write(f"模型输出: {pred_text}\n")
                f.write("-" * 80 + "\n")
        
        f.write(f"\n总计错误样本数: {error_count}\n")
    print(f"✅ 错误样本已保存到: {eval_error_path}")

    # ===================== 🚀 自动生成评估图片 =====================
    plot_metric_bar_chart(cls_report)
    plot_confusion_matrix(test_true_labels, test_pred_labels)

    print(f"\n🎯 推理样例展示（前3条）：")
    print("-" * 80)
    for i in range(min(3, len(test_payloads))):
        print(f"🔹 真实标签：{test_true_labels[i]}")
        print(f"🔹 预测标签：{test_pred_labels[i]}")
        print(f"✅ 生成研判：{test_pred_texts[i]}")
        print("-" * 80)

    print("=" * 80)
    print("🎉 全部完成！")
    print(f"📊 评估结果: {eval_result_path}")
    print(f"📊 错误记录: {eval_error_path}")
    print("=" * 80)