import sys
import os
import time
import torch
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# 镜像（不影响）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# 加载模型
model_path = "./pythia-70m-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPTNeoXForCausalLM.from_pretrained(model_path).cuda()
model.eval()

# 配置
MAX_LENGTH = 128
GEN_LEN = 5  # 从 1 改成 5 → 关键修复！
LABELS = ["cmdi", "normal", "sqli", "xss"]
label_map = {"cmdi":"cmdi", "normal":"normal", "sqli":"sqli", "xss":"xss"}

# 数据集
from dataset import AttackDataset
class DummyVocab:
    def __getitem__(self, x): return 0
    def __len__(self): return 100

dummy = DummyVocab()
test_dataset = AttackDataset(dummy, mode="test")
total_samples = len(test_dataset.raw_samples)

# ==================== 预测函数（已修复） ====================
def predict(payload):
    prompt = f"PAYLOAD:{payload} LABEL:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_LEN,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    # 最强匹配：只要出现关键词就返回
    if "cmdi" in text:
        return "cmdi"
    elif "normal" in text:
        return "normal"
    elif "sqli" in text:
        return "sqli"
    elif "xss" in text:
        return "xss"
    else:
        return "normal"  # 兜底，保证永远返回4类之一

# ==================== 预热 + 计时推理 ====================
y_true = []
y_pred = []

# GPU 预热，消除冷启动误差
warmup_cnt = 10
with torch.no_grad():
    for txt, _ in list(zip(test_dataset.raw_samples, test_dataset.augmented_type_list))[:warmup_cnt]:
        _ = predict(txt)

# 正式计时开始
start_time = time.perf_counter()

for txt, lab in tqdm(zip(test_dataset.raw_samples, test_dataset.augmented_type_list), total=total_samples):
    y_true.append(lab)
    y_pred.append(predict(txt))

# 计时结束
end_time = time.perf_counter()

# 计算推理性能指标
total_infer_time_s = end_time - start_time
avg_latency_ms = (total_infer_time_s / total_samples) * 1000
throughput = total_samples / total_infer_time_s

# ==================== 评估指标计算 ====================
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4, target_names=LABELS)
total_params = sum(p.numel() for p in model.parameters())
model_size = total_params * 4 / 1024 / 1024

# ==================== 控制台输出 ====================
print("\n" + "="*80)
print("Pythia-70M 模型评估结果")
print("="*80)
print(f"整体准确率: {acc:.4f}\n")
print(report)
print(f"\n模型总参数量: {total_params:,}")
print(f"模型大小: {model_size:.2f} MB")
print(f"总测试样本数: {total_samples}")
print(f"推理总耗时: {total_infer_time_s:.2f} s")
print(f"单样本平均推理时延: {avg_latency_ms:.2f} ms/sample")
print(f"推理吞吐量: {throughput:.2f} samples/s\n")

# ==================== 保存到日志文件 ====================
with open("pythia-evaluation-result.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("Pythia-70M 模型评估结果\n")
    f.write("="*80 + "\n")
    f.write(f"整体准确率: {acc:.4f}\n\n")
    f.write(report)
    f.write(f"\n模型总参数量: {total_params:,}\n")
    f.write(f"模型大小: {model_size:.2f} MB\n")
    f.write(f"总测试样本数: {total_samples}\n")
    f.write(f"推理总耗时: {total_infer_time_s:.2f} 秒\n")
    f.write(f"单样本平均推理时延: {avg_latency_ms:.2f} 毫秒/样本\n")
    f.write(f"推理吞吐量: {throughput:.2f} 样本/秒\n")

print("✅ 评估结果已保存至 pythia-evaluation-result.txt")