import sys, os
import time
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ==================== T5 模型=====================
tokenizer = T5Tokenizer.from_pretrained("t5-final-valid")
model = T5ForConditionalGeneration.from_pretrained("t5-final-valid").cuda()
model.eval()

# ==================== dataset.py ====================
from dataset import AttackDataset

# 实例化 AttackDataset
class DummyVocab:
    def __getitem__(self, x): return 0
    def __len__(self): return 100

vocab_placeholder = DummyVocab()
test_dataset = AttackDataset(vocab_placeholder, mode="test")

# 直接拿数据集里的原始 payload 和 真实标签
payload_list = test_dataset.raw_samples
y_true = test_dataset.augmented_type_list
total_samples = len(payload_list)

# ==================== 推理 + 计时统计 ====================
y_pred = []
# 模型预热，消除GPU冷启动耗时（关键）
warmup_cnt = 10
with torch.no_grad():
    for payload in payload_list[:warmup_cnt]:
        inputs = tokenizer(
            payload,
            max_length=96,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids.cuda()
        _ = model.generate(inputs, max_length=8)

# 正式开始计时
start_time = time.perf_counter()
with torch.no_grad():
    for payload in tqdm(payload_list, desc="Evaluating"):
        inputs = tokenizer(
            payload,
            max_length=96,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids.cuda()

        outputs = model.generate(inputs, max_length=8)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        y_pred.append(pred)
# 结束计时
end_time = time.perf_counter()

# 计算推理指标
total_infer_time_s = end_time - start_time          # 总推理耗时(秒)
avg_latency_ms = (total_infer_time_s / total_samples) * 1000  # 单样本平均时延(毫秒)
throughput = total_samples / total_infer_time_s    # 吞吐量(样本/秒)

# ==================== 计算分类指标 ====================
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)

# ==================== 模型大小、参数量 ====================
total_params = sum(p.numel() for p in model.parameters())
model_size_mb = total_params * 4 / (1024 ** 2)

# ==================== 控制台输出结果 ====================
print("\n" + "=" * 88)
print("T5-Tiny 模型评估结果（使用 dataset.py | 全量测试集 6338 条）")
print("=" * 88)
print(f"整体准确率: {acc:.4f}\n")
print(report)
print(f"\n模型总参数量: {total_params:,}")
print(f"模型大小: {model_size_mb:.2f} MB")
print(f"总测试样本数: {total_samples}")
print(f"推理总耗时: {total_infer_time_s:.2f} s")
print(f"单样本平均推理时延: {avg_latency_ms:.2f} ms/sample")
print(f"推理吞吐量: {throughput:.2f} samples/s")

# ==================== 保存结果到日志文件 ====================
with open("t5-tiny-evaluation-result.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("T5-Tiny 模型评估结果\n")
    f.write("=" * 80 + "\n")
    f.write(f"整体准确率: {acc:.4f}\n\n")
    f.write(report)
    f.write(f"\n\n模型总参数量: {total_params:,}\n")
    f.write(f"模型大小: {model_size_mb:.2f} MB\n")
    f.write(f"总测试样本数: {total_samples}\n")
    f.write(f"推理总耗时: {total_infer_time_s:.2f} 秒\n")
    f.write(f"单样本平均推理时延: {avg_latency_ms:.2f} 毫秒/样本\n")
    f.write(f"推理吞吐量: {throughput:.2f} 样本/秒\n")

print(f"\n✅ 评估结果已保存至 t5-tiny-evaluation-result.txt")