# ===================================
# Common.py        
# 负责公共参数、函数管理
# ===================================

import re
import torch
import os

# 设置环境变量，避免显存占用过大，意思是设置显存占用为4096M
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
#================================== 路径 ===============================
# 程序根目录
rootpath ="./workspace/webattackgpt_dictionary/"
# 词表路径
vocab_path = f"{rootpath}vocab.pth"
# 模型路径
model_path = f"{rootpath}gpt_model.pth"
# 数据集路径
dataset_path = f"{rootpath}csicdataset/"
# 损失曲线图路径
loss_curve_path = f"{rootpath}loss_curve.png"
# 指标柱状图路径
metric_bar_chart_path = f"{rootpath}metric_bar_chart.png"
# 混淆矩阵图路径
confusion_matrix_path = f"{rootpath}confusion_matrix.png"
# 评估结果路径
eval_result_path = f"{rootpath}eval_result.txt"
# 评估错误结果
eval_error_path = f"{rootpath}eval_error.txt"

#================================== 模型参数 ===============================
# 解码器层数
n_layers = 3  
# 隐藏层大小
hidden_size = 256
# 词嵌入大小
d_embedding = 128
# 注意力头数
n_heads = 2 
# 注意力键K、值V维度
d_k = d_v = 64
# 批大小
batch_size = 16
# 训练轮数
epochs = 110 
# 最大序列长度
max_seq_len = 512 
# dropout
dropout = 0.05
# 早停轮数
early_stop_patience = 5
# 早停计数器
early_stop_counter = 0
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 特殊token
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "<seq1>", "<seq2>"]
# 特殊token正则表达式
SPECIAL_PATTERN = re.compile('|'.join(re.escape(t) for t in SPECIAL_TOKENS))
# 词表兜底字符
SAFE_SYMBOLS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SAFE_SYMBOLS += "<>()'\",.?/&=;_+-[]|\\*@#%^~` "
# 词表中文字符
CHINESE = "请求内容研判结果该Payload为正常流量无攻击行为风险等级低高xsssqlicmdi"

#================================== 公共函数 ===============================
# 分词函数：特殊字符<sos><eos>等优先，再以单个中英文字符作为token，否则将其标为<unk>
def tokenize(text):
  tokens = []
  i = 0
  n = len(text)
  # 从左到右，一个一个扫描字符
  while i < n:
      # 第一步：检查是不是特殊token
      matched = None
      # 遍历所有特殊token
      for st in SPECIAL_TOKENS:
        # 从位置 i 开始，是不是以这个特殊token开头
        if text.startswith(st, i):
            matched = st
            break
      # 如果匹配到了特殊token
      if matched:
        # 直接把整个特殊token加进去
        tokens.append(matched)
        # 跳过整个特殊token的长度
        i += len(matched)
        # 继续扫描下一段
        continue
      # 第二步：不是特殊token，按单个字符处理
      # 拿到当前这个字符
      c = text[i]
      # 如果是【安全符号】或【中文】，直接保留
      if c in SAFE_SYMBOLS or c in CHINESE:
        tokens.append(c)
      else:
        # 否则 → 不认识 → 标 <unk>
        tokens.append("<unk>")
      i += 1
  return tokens

# 先截断再填充
def pad_sequence(sequences, padding_value, max_len):
  # 超长先截断！保证所有序列 ≤ max_len
  sequences = [seq[:max_len] for seq in sequences]
  # 再统一填充到 max_len
  batch_size = len(sequences)
  result = torch.full((batch_size, max_len), padding_value, dtype=torch.long)
  for i, seq in enumerate(sequences):
      result[i, :len(seq)] = seq
  return result

# 强制对齐max_seq_len, 用<pad>填充
def collate_fn(batch, vocab):
  src = [item[0] for item in batch]
  tgt = [item[1] for item in batch]
  
  src_padded = pad_sequence(src, padding_value=vocab["<pad>"], max_len=max_seq_len)
  tgt_padded = pad_sequence(tgt, padding_value=vocab["<pad>"], max_len=max_seq_len)
  
  return src_padded.to(device), tgt_padded.to(device)

# 动态计算最大生成长度
def generate_detection(payload, model, vocab):
  # 构造Prompt（和训练完全一致）
  prompt = f"<sos> 请求内容:{payload} <seq1> 研判结果:"
  
  # 分词
  tokens = tokenize(prompt)
  input_ids = [vocab[t] for t in tokens]
  input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

  # 计算：还能生成多少个token
  prompt_token_len = len(tokens)
  max_generate_len = max_seq_len - prompt_token_len

  # 自回归生成
  with torch.no_grad():
    for _ in range(max_generate_len):
      # 防止输入超过模型最大长度
      input_tensor = input_tensor[:, -max_seq_len:]
      # 模型生成
      outputs = model(input_tensor)
      # outputs.argmax(-1)获取[序列长度, 词表大小]最后一维的最大概率索引
      next_token_id = outputs.argmax(-1)[:, -1].item()
      # 将生成的token追加到输入中
      input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]]).to(device)], dim=-1)
      
      if next_token_id == vocab["<eos>"]:
        break
              
  # 转换文本
  result_tokens = [vocab.get_itos()[idx] for idx in input_tensor[0].tolist()]
  # 去除特殊token
  pure_result = "".join([t for t in result_tokens if t not in SPECIAL_TOKENS])
  
  return pure_result