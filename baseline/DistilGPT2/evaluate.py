import sys, os
import time
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== DistilGPT2 ====================
tokenizer = AutoTokenizer.from_pretrained("distilgpt2-final")
model = AutoModelForCausalLM.from_pretrained("distilgpt2-final").cuda()
model.eval()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ==================== dataset.py ====================
from dataset import AttackDataset

class DummyVocab:
    def __getitem__(self, x): return 0
    def __len__(self): return 100

vocab_placeholder = DummyVocab()
test_dataset = AttackDataset(vocab_placeholder, mode="test")

payload_list = test_dataset.raw_samples
y_true = test_dataset.augmented_type_list
total_samples = len(payload_list)

# ==================== 预测 + 计时统计 ====================
y_pred = []

# GPU 预热，消除冷启动偏差
warmup_cnt = 10
with torch.no_grad():
    for payload in payload_list[:warmup_cnt]:
        text = f"PAYLOAD: {payload} LABEL:"
        inputs = tokenizer(text, return_tensors="pt", max_length=96, truncation=True).input_ids.cuda()
        _ = model.generate(inputs, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)

# 正式开始计时
start_time = time.perf_counter()
with torch.no_grad():
    for payload in tqdm(payload_list):
        text = f"PAYLOAD: {payload} LABEL:"
        inputs = tokenizer(text, return_tensors="pt", max_length=96, truncation=True).input_ids.cuda()
        outputs = model.generate(inputs, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)
        res = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        
        pred = "normal"
        for l in ["cmdi", "normal", "sqli", "xss"]:
            if l in res:
                pred = l
                break
        y_pred.append(pred)
# 计时结束
end_time = time.perf_counter()

# 计算推理性能指标
total_infer_time_s = end_time - start_time
avg_latency_ms = (total_infer_time_s / total_samples) * 1000
throughput = total_samples / total_infer_time_s

# ==================== 指标 ====================
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4, zero_division=0)

# 模型参数量 & 大小
total_params = sum(p.numel() for p in model.parameters())
model_size = total_params * 4 / 1024 / 1024

# ==================== 控制台输出 ====================
print("\n" + "=" * 88)
print("DistilGPT2 评估结果（全量测试集 6338）")
print("=" * 88)
print(f"整体准确率: {acc:.4f}\n")
print(report)
print(f"模型参数量: {total_params:,}")
print(f"模型大小: {model_size:.2f} MB")
print(f"总测试样本数: {total_samples}")
print(f"推理总耗时: {total_infer_time_s:.2f} s")
print(f"单样本平均推理时延: {avg_latency_ms:.2f} ms/sample")
print(f"推理吞吐量: {throughput:.2f} samples/s\n")

# ==================== 保存日志文件 ====================
with open("distilgpt2-result.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("DistilGPT2 Evaluation Result\n")
    f.write("=" * 80 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)
    f.write(f"\n\nParams: {total_params:,}\n")
    f.write(f"Size: {model_size:.2f} MB\n")
    f.write(f"Test Samples: {total_samples}\n")
    f.write(f"Total Inference Time: {total_infer_time_s:.2f} s\n")
    f.write(f"Average Latency: {avg_latency_ms:.2f} ms/sample\n")
    f.write(f"Throughput: {throughput:.2f} samples/s\n")

print("✅ 评估结果已保存至 distilgpt2-result.txt")