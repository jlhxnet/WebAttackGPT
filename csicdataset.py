# ===================================
# csicdataset.py
# 负责数据集管理
# ===================================

from torch.utils.data import Dataset
from common import *
import torch
import pandas as pd
import os
from collections import Counter

class AttackDataset(Dataset):
    def __init__(self, vocab, mode="train"):
        # 字典
        self.vocab = vocab
        # 数据集模式，train/valid/test
        self.mode = mode
        self.data = []
        # 记录【扩增后】每条样本的类型
        self.augmented_type_list = []
        # 组合得到数据集位置
        csv_path = os.path.join(dataset_path, f"{mode}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据集不存在: {csv_path}")
        # 读取数csv文件得到数据集,并去除空行
        self.df = pd.read_csv(csv_path).dropna()

        # 遍历原始数据，生成并扩增样本
        for _, row in self.df.iterrows():
            try:
                # 读取原始样本payload部分并去掉空格
                payload = str(row["payload"]).strip()
                # 读取原始样本类型部分并去掉空格，并转换为小写
                atk_type = str(row["attack_type"]).strip().lower()
                
                # 生成研判文本
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

                # 组装序列
                seq = f"<sos> 请求内容:{payload} <seq1> 研判结果:{judge_text} <eos>"
                # 词元化
                tokens = tokenize(seq)
                # 构造src和tgt，src去掉<eos>, tgt去掉<sos>
                src = torch.tensor([vocab[t] for t in tokens[:-1]])
                tgt = torch.tensor([vocab[t] for t in tokens[1:]])

                # 添加原始样本
                self.data.append((src, tgt))
                self.augmented_type_list.append(atk_type)  # 记录类型

                # 如果训练，还需要扩增
                if mode == "train":
                    if atk_type == "cmdi":
                        # 1倍扩增CMDI样本
                        for _ in range(1):
                            self.data.append((src, tgt))
                            self.augmented_type_list.append(atk_type)
                    elif atk_type == "sqli":
                        # 2倍扩增sqli样本
                        for _ in range(2):
                            self.data.append((src, tgt))
                            self.augmented_type_list.append(atk_type)
                    elif atk_type == "xss":
                        # 3倍扩增xss样本
                        for _ in range(3):
                            self.data.append((src, tgt))
                            self.augmented_type_list.append(atk_type)
                    # normal无扩增
                    elif atk_type == "normal":
                        pass
            except Exception:
                continue

        # 如果训练，统计各类别数量
        if mode == "train":
            print("\n" + "="*60)
            print(f"📊 【训练集】")
            print("="*60)
            # 统计训练样本. Counter的作用是统计每个类别的数量
            class_counts = Counter(self.augmented_type_list)
            # 打印各类别数量，sorted()函数用于排序，key参数指定排序的键，reverse参数指定是否反向排序
            for atk_type, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
              # f-string格式化字符串，<:15表示左对齐15个字符，>:6表示右对齐6个字符
              print(f"🔹 {atk_type:<15} : {count:>6} 条")
            print(f"📌 训练集总样本数 : {len(self.data):>6} 条")
            print("="*60 + "\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # 为数据集生成词元
    @classmethod
    def yield_attack_tokens(cls):
        # 先喂所有兜底符号
        yield list(SAFE_SYMBOLS)
        # 再喂所有中文符号
        yield list(CHINESE)
        
        # 遍历所有数据，含train/valid/test
        for mode in ["train", "valid", "test"]:
            p = os.path.join(dataset_path, f"{mode}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    try:
                        payload = str(r["payload"]).strip()
                        atk_type = r["attack_type"].strip().lower()
                        # 训练序列
                        if atk_type == "normal":
                            jg = "该Payload为正常流量，无攻击行为，风险等级低"
                        elif atk_type == "xss":
                            jg = "该Payload为xss攻击，尝试窃取网页权限，风险等级高"
                        elif atk_type == "sqli":
                            jg = "该Payload为sqli攻击，尝试窃取数据库数据，风险等级高"
                        elif atk_type == "cmdi":
                            jg = "该Payload为cmdi命令注入攻击，尝试窃取网页权限，风险等级高"
                        else:
                            jg = "该Payload为未知流量，需人工核查，风险等级中"
                        
                        seq = f"<sos> 请求内容:{payload} <seq1> 研判结果:{jg} <eos>"
                        # 词元化
                        yield tokenize(seq)
                    except:
                        continue