import sys, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==================== T5 模型 ====================
tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny").cuda()

# ==================== 超参数 ====================
BATCH_SIZE = 2
EPOCHS = 6
LR = 1e-4
MAX_LENGTH = 96

# ==================== dataset.py ====================
from dataset import AttackDataset

# 实例化 AttackDataset
class DummyVocab:
    def __getitem__(self, x): return 0
    def __len__(self): return 100

vocab = DummyVocab()
train_dataset = AttackDataset(vocab, mode="train")

payloads = []
labels = []
for atk_type, payload in zip(train_dataset.augmented_type_list, train_dataset.raw_samples):
    payloads.append(payload)
    labels.append(atk_type)

# ==================== 构造训练集 ====================
from torch.utils.data import TensorDataset
class SimpleDataset(TensorDataset):
    def __init__(self):
        self.x = payloads
        self.y = labels
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_ds = SimpleDataset()

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    inp = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    lab = tokenizer(labels, max_length=8, truncation=True, padding="max_length", return_tensors="pt")
    lab["input_ids"][lab["input_ids"] == tokenizer.pad_token_id] = -100

    return inp["input_ids"].cuda(), lab["input_ids"].cuda()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==================== 训练 ====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for src, tgt in tqdm(train_loader):
        loss = model(input_ids=src, labels=tgt).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    print(f"Epoch {epoch+1} | loss: {total/len(train_loader):.3f}")

# ==================== 保存====================
model.save_pretrained("t5-final-valid")
tokenizer.save_pretrained("t5-final-valid")

print("\n✅ 训练完成！")