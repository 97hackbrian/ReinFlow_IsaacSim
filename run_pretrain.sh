#!/bin/bash
export REINFLOW_DIR=/workspace/ReinFlow
export REINFLOW_DATA_DIR=/workspace/ReinFlow/data
export REINFLOW_LOG_DIR=/workspace/ReinFlow/log
export REINFLOW_WANDB_ENTITY=null
export PYTHONPATH=/workspace/ReinFlow
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd /workspace/ReinFlow
python3 script/run.py --config-dir cfg/isaac/pretrain/xarm_screwdriver --config-name pre_shortcut_mlp wandb.offline_mode=True train.n_epochs=200
