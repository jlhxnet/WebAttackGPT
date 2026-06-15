import sys
import os

# 使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from tqdm import tqdm

# ==================== 超参数 ====================
BATCH_SIZE = 4
EPOCHS = 12
LR = 2e-5
MAX_LENGTH = 128

# ==================== 从国内镜像下载 Pythia-70M ====================
model_name = "EleutherAI/pythia-70m-deduped"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name).cuda()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ==================== 数据集 ====================
from dataset import AttackDataset

class DummyVocab:
    def __getitem__(self, x): return 0
    def __len__(self): return 100

dummy = DummyVocab()
train_dataset = AttackDataset(dummy, mode="train")

# 纯生成式格式
class WebGenDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = [f"PAYLOAD:{t} LABEL:{l}" for t, l in zip(texts, labels)]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

ds = WebGenDataset(train_dataset.raw_samples, train_dataset.augmented_type_list)

def collate_fn(batch):
    tok = tokenizer(
        batch, max_length=MAX_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    input_ids = tok["input_ids"].cuda()
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, labels

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ==================== 训练 ====================
model.train()
print("\n✅ Pythia-70M 训练开始 \n")

for epoch in range(EPOCHS):
    total_loss = 0.0
    for x, y in tqdm(loader):
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.3f}")

# ==================== 保存 ====================
model.save_pretrained("pythia-70m-final")
tokenizer.save_pretrained("pythia-70m-final")

total_params = sum(p.numel() for p in model.parameters())
model_size = total_params * 4 / 1024 / 1024
print(f"\n🎉 训练完成！")
print(f"模型参数量: {total_params:,}")
print(f"模型大小: {model_size:.2f} MB")