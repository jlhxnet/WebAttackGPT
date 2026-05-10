# WebAttackGPT
Dictionary-Enhanced Lightweight GPT for Generative Judgment of Web Attack Intent

## Overview
WebAttackGPT is an ultra-lightweight generative model for Web attack detection and intent judgment. It is designed for edge deployment with only 0.61M parameters and provides interpretable detection results.

The model integrates a novel AttentionDictionary module to enhance high-risk malicious features, suppress irrelevant noise, and significantly improve detection performance on long-tailed attacks such as CMDI, SQLi, and XSS.

## Key Features
- Ultra-lightweight GPT decoder backbone (3 layers, 2 attention heads)
- AttentionDictionary: learnable feature prototype enhancement for attack payloads
- Global high-risk token recognition (CMDI / XSS / SQLi)
- Character-level HTTP payload tokenization
- End-to-end generative attack intent reasoning
- Strong performance on the public CSIC-2010 dataset

## Project Structure
WebAttackGPT/
├── attentiondictionary.py       # Core dictionary-enhanced attention module
├── common.py                     # Global config, paths, tokenizer, utilities
├── csicdataset.py                # Attack dataset loader and augmentation
├── csicpreprocess.py             # Raw dataset parsing & train/valid/test split
├── decoder.py                    # GPT decoder with positional encoding
├── decoderlayer.py               # Single decoder layer with pre-norm structure
├── dictenhancedblock.py          # Feature enhancement block wrapper
├── evaluate.py                   # Model evaluation (Acc, F1, Recall, confusion matrix)
├── gpt.py                        # Full lightweight GPT model definition
├── loadmodel.py                  # Model & vocabulary loading interface
├── multiheadattention.py         # Multi-head causal self-attention
├── poswisefeedforwardnet.py      # Position-wise feed-forward network (Conv1D)
├── scaleddotproductattention.py  # Scaled dot-product attention core
├── test.py                       # Inference demo for attack payload detection
├── train.py                      # Model training pipeline with early stopping
├── requirements.txt              # Dependencies
└── README.md                     # Project description

## Requirements
torch>=2.0.0
numpy>=121.0
pandas>=1.4.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
tqdm>=4.62.0
torchtext>=0.15.0

## Quick Start
1. Install dependencies
pip install -r requirements.txt

2. Preprocess the raw CSIC-2010 dataset
python csicpreprocess.py

3. Build vocabulary and train the model
python train.py

4. Evaluate model performance
python evaluate.py

5. Run inference demo
python test.py

## Dataset
All experiments are conducted on the CSIC-2010 HTTP Web Attack Dataset.
Official link: https://www.isi.csic.es/dataset/

## Performance (CSIC-2010)
- Accuracy: 96.84%
- Macro-F1: 97.40%
- CMDI Recall: 91.41%
- Total Parameters: 0.61M

## Citation
If you use this code for academic research, please cite our paper:

Li Jiao, Muhammad Irsyad Abdullah, Limei Zhao, Jiao Zhao, Bin Tang.
Dictionary-Enhanced Lightweight GPT for Generative Judgment of Web Attack Intent.
Computer Networks, 2026.

## Usage Restrictions
This code is ONLY for academic reference and learning.
**ANY unauthorized use of this code, model, or structure to write, submit, or publish academic papers is strictly prohibited.**
All rights of the original algorithm and code belong to the authors.

## Copyright
This code is released for academic research only.
Copyright © 2026 The Authors. All Rights Reserved.