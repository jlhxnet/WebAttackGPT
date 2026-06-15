from torch.utils.data import Dataset
from common import *
import torch
import os
import random
from collections import Counter

# ===================================
#          负责数据集管理（CSIC+OWASP混合稳定版）
# ===================================
class AttackDataset(Dataset):
    def __init__(self, vocab, mode="train"):
        self.vocab = vocab
        self.mode = mode
        self.data = []
        self.augmented_type_list = []
        self.raw_samples = []  # 保存原始 payload
        all_samples = []

        # ========== 1、读取 CSIC 数据集 ==========
        csv_path = os.path.join(dataset_path, f"{mode}.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                header = lines[0].strip()
                for line in lines[1:]:
                    line = line.strip()
                    if not line or line.count(",") < 2:
                        continue
                    parts = line.split(",", 2)
                    if len(parts) != 3:
                        continue
                    payload = parts[0].strip()
                    atk_type = parts[1].strip().lower()
                    all_samples.append(("csv", payload, atk_type))
            except:
                pass

        # ========== 2、读取 OWASP 数据集 ==========
        owasp_txt = os.path.join(dataset_path, f"owasp_{mode}.txt")
        if os.path.exists(owasp_txt):
            with open(owasp_txt, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.count(",") < 2:
                        continue
                    try:
                        payload, atk_type, _ = line.rsplit(",", 2)
                        payload = payload.strip()
                        atk_type = atk_type.strip().lower()
                        all_samples.append(("txt", payload, atk_type))
                    except:
                        continue

        if len(all_samples) == 0:
            raise FileNotFoundError(f"❌ {mode} 数据集为空！")

        # 训练集打乱
        if mode == "train":
            random.seed(42)
            random.shuffle(all_samples)

        # ========== 构建样本 ==========
        for _, payload, atk_type in all_samples:
            try:
                if atk_type == "normal":
                    judge_text = "该Payload为正常流量，无攻击行为，风险等级低"
                elif atk_type == "xss":
                    judge_text = "该Payload为xss攻击，尝试窃取网页权限，风险等级高"
                elif atk_type == "sqli":
                    judge_text = "该Payload为sqli攻击，尝试窃取数据库数据，风险等级高"
                elif atk_type == "cmdi":
                    judge_text = "该Payload为cmdi命令注入攻击，尝试执行系统命令，风险等级高"
                else:
                    judge_text = "该Payload为未知流量，需人工核查，风险等级中"

                seq = f"<sos> 请求内容:{payload} <seq1> 研判结果:{judge_text} <eos>"
                tokens = tokenize(seq)
                src = torch.tensor([vocab[t] for t in tokens[:-1]])
                tgt = torch.tensor([vocab[t] for t in tokens[1:]])

                # 训练集扩增，测试集不扩增
                if mode == "train":
                    self.data.append((src, tgt))
                    self.augmented_type_list.append(atk_type)
                    self.raw_samples.append(payload)

                    # ===================== 优化扩增：复制 + 轻量扰动增强 =====================
                    if atk_type == "cmdi":
                        for _ in range(1):
                            new_payload = self.augment_payload(payload)
                            seq_aug = f"<sos> 请求内容:{new_payload} <seq1> 研判结果:{judge_text} <eos>"
                            tokens_aug = tokenize(seq_aug)
                            src_aug = torch.tensor([vocab[t] for t in tokens_aug[:-1]])
                            tgt_aug = torch.tensor([vocab[t] for t in tokens_aug[1:]])
                            self.data.append((src_aug, tgt_aug))
                            self.augmented_type_list.append(atk_type)
                            self.raw_samples.append(new_payload)

                    elif atk_type == "sqli":
                        for _ in range(2):
                            new_payload = self.augment_payload(payload)
                            seq_aug = f"<sos> 请求内容:{new_payload} <seq1> 研判结果:{judge_text} <eos>"
                            tokens_aug = tokenize(seq_aug)
                            src_aug = torch.tensor([vocab[t] for t in tokens_aug[:-1]])
                            tgt_aug = torch.tensor([vocab[t] for t in tokens_aug[1:]])
                            self.data.append((src_aug, tgt_aug))
                            self.augmented_type_list.append(atk_type)
                            self.raw_samples.append(new_payload)

                    elif atk_type == "xss":
                        # XSS 扩增 6 倍（5次增强 + 1次原始）
                        for _ in range(5):
                            new_payload = self.augment_payload(payload)
                            seq_aug = f"<sos> 请求内容:{new_payload} <seq1> 研判结果:{judge_text} <eos>"
                            tokens_aug = tokenize(seq_aug)
                            src_aug = torch.tensor([vocab[t] for t in tokens_aug[:-1]])
                            tgt_aug = torch.tensor([vocab[t] for t in tokens_aug[1:]])
                            self.data.append((src_aug, tgt_aug))
                            self.augmented_type_list.append(atk_type)
                            self.raw_samples.append(new_payload)
                    # ==========================================================================

                else:
                    # 测试集/验证集只添加一次
                    self.data.append((src, tgt))
                    self.augmented_type_list.append(atk_type)
                    self.raw_samples.append(payload)

            except:
                continue

        # 打印训练集统计
        if mode == "train":
            print("\n" + "="*60)
            print(f"📊 训练集 | CSIC + OWASP 融合完成")
            print("="*60)
            cnt = Counter(self.augmented_type_list)
            for k, v in sorted(cnt.items(), key=lambda x: x[1], reverse=True):
                print(f"🔹 {k:<12} : {v:>6} 条")
            print(f"✅ 训练集总样本 : {len(self.data)} 条")
            print("="*60 + "\n")

    # 轻量Payload增强：不改变攻击语义，仅做混淆
    def augment_payload(self, payload):
        aug_payload = list(payload)
        aug_type = random.choice(["none", "space", "upper_lower"])
        if aug_type == "space" and len(aug_payload) > 2:
            insert_pos = random.sample(range(len(aug_payload)), min(1, len(aug_payload)))
            for pos in sorted(insert_pos, reverse=True):
                aug_payload.insert(pos, " ")
        elif aug_type == "upper_lower":
            for i in range(len(aug_payload)):
                c = aug_payload[i]
                if c.isalpha():
                    aug_payload[i] = c.upper() if random.random() > 0.5 else c.lower()
        return "".join(aug_payload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ===================== 词表生成 =====================
    @classmethod
    def yield_attack_tokens(cls):
        yield list(SAFE_SYMBOLS)
        yield list(CHINESE)

        for mode in ["train", "valid", "test"]:
            p = os.path.join(dataset_path, f"{mode}.csv")
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for line in lines[1:]:
                        line = line.strip()
                        if not line or line.count(",") < 2:
                            continue
                        payload, atk_type, _ = line.split(",", 2)
                        payload = payload.strip()
                        atk_type = atk_type.strip().lower()
                        jg = ""
                        if atk_type == "normal":
                            jg = "该Payload为正常流量，无攻击行为，风险等级低"
                        elif atk_type == "xss":
                            jg = "该Payload为xss攻击，尝试窃取网页权限，风险等级高"
                        elif atk_type == "sqli":
                            jg = "该Payload为sqli攻击，尝试窃取数据库数据，风险等级高"
                        elif atk_type == "cmdi":
                            jg = "该Payload为cmdi命令注入攻击，尝试执行系统命令，风险等级高"
                        else:
                            jg = "该Payload为未知流量，需人工核查，风险等级中"
                        seq = f"<sos> 请求内容:{payload} <seq1> 研判结果:{jg} <eos>"
                        yield tokenize(seq)
                except:
                    continue

            p = os.path.join(dataset_path, f"owasp_{mode}.txt")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.count(",") < 2:
                            continue
                        try:
                            payload, atk_type, _ = line.rsplit(",", 2)
                            payload = payload.strip()
                            atk_type = atk_type.strip().lower()
                            jg = ""
                            if atk_type == "normal":
                                jg = "该Payload为正常流量，无攻击行为，风险等级低"
                            elif atk_type == "xss":
                                jg = "该Payload为xss攻击，尝试窃取网页权限，风险等级高"
                            elif atk_type == "sqli":
                                jg = "该Payload为sqli攻击，尝试窃取数据库数据，风险等级高"
                            elif atk_type == "cmdi":
                                jg = "该Payload为cmdi命令注入攻击，尝试执行系统命令，风险等级高"
                            else:
                                jg = "该Payload为未知流量，需人工核查，风险等级中"
                            seq = f"<sos> 请求内容:{payload} <seq1> 研判结果:{jg} <eos>"
                            yield tokenize(seq)
                        except:
                            continue