import sys, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==================== DistilGPT2 自带词表 ====================
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").cuda()
tokenizer.pad_token = tokenizer.eos_token

# ==================== 超参 ====================
BATCH_SIZE = 2
EPOCHS = 12
LR = 1e-4
MAX_LENGTH = 96

# ==================== dataset.py ====================
from dataset import AttackDataset

class DummyVocab:
    def __getitem__(self, x): return 0
    def __len__(self): return 100

vocab = DummyVocab()
train_dataset = AttackDataset(vocab, mode="train")

# 原始 payload + label
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.pairs = list(zip(train_dataset.raw_samples, train_dataset.augmented_type_list))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

ds = SimpleDataset()

def collate_fn(batch):
    texts = [f"payload:{p} label:{l}" for p, l in batch]
    tok = tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    tok["labels"] = tok["input_ids"]
    return tok.input_ids.cuda(), tok.attention_mask.cuda(), tok.labels.cuda()

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ==================== 训练 ====================
model.train()
for epoch in range(EPOCHS):
    total = 0
    for input_ids, attn, labels in tqdm(loader):
        loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    print(f"Epoch {epoch+1} | loss: {total/len(loader):.3f}")

model.save_pretrained("distilgpt2-final")
tokenizer.save_pretrained("distilgpt2-final")
print("\n✅ DistilGPT2 训练完成")