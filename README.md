# WebAttackGPT

## Copyright & Usage Notice
Copyright (c) 2025. All rights reserved.
This source code is provided **solely for technical exchange, learning, and research purposes only**.
Unauthorized use, reproduction, modification, or distribution of this code, in whole or in part, for academic publication, commercial use, or any other form of public dissemination is strictly prohibited without explicit written permission from the author.

---

## Environment Setup
Install all dependencies with:
pip install -r requirements.txt

## Project Structure
- baseline/: Implementation of baseline comparison models
- dataset/ : Dataset files
- Preprocessing: `csicpreprocess.py`, `owasappreprocess.py`
- Core modules: `gpt.py`, `decoderlayer.py`, `multihadattention.py`, `attentiondictionary.py`, `dictenhancedblock.py`, `poswisefeedforwardnet.py`, `scaleddotproductatention.py`
- Utilities: `common.py`, `load_model.py`
- Main scripts:
  - `train.py`: Model training pipeline
  - `evaluate.py`: Standard model evaluation
  - `evaluate_AARE.py`: AARE metric-specific evaluation

## Usage
1. Model Training
python train.py

2. Model Evaluation
python evaluate.py
python evaluate_AARE.py

3. Baseline Experiments
Navigate into the baseline folder and run the corresponding scripts to reproduce baseline experimental results.

## Notes
- You can regenerate new logs and figures by executing the code.
- Trained model weights are not included in this package.
