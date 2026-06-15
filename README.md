WebAttackGPT
Source code for the paper submitted to PeerJ.

Environment Setup
Install all dependencies with:
pip install -r requirements.txt

Project Structure
- baseline/: Implementation of all baseline comparison models
- dataset/, csicdataset/: Dataset files and related resources
- csicpreprocess.py, owasappreprocess.py: Data preprocessing scripts
- gpt.py, decoderlayer.py, multihadattention.py: Core modules of the proposed model
- common.py, dataset.py, load_model.py: General utility functions

Usage
1. Model Training
python train.py

2. Model Evaluation
python evaluate.py
python evaluate_AARE.py

3. Baseline Experiments
Navigate into the baseline folder and run the corresponding scripts to reproduce baseline experimental results.

Notes
- Existing log files and generated figures are demonstration results from local runs. You can regenerate new logs and figures by executing the code.
- Trained model weights are not included in this package.
- All experiments in this paper can be fully reproduced with the provided code.
