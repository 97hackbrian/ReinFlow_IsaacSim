#!/bin/bash
docker exec -it reinflow-mujoco-container bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate reinflow && source script/setup_env.sh && python script/run.py --config-dir cfg/isaac/pretrain/xarm_screwdriver --config-name pre_shortcut_mlp train.n_epochs=200"
