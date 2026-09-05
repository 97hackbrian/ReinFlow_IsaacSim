#!/bin/bash
LAST_CKPT=$(ls -td log/isaac/pretrain/xarm_screwdriver/*/*/checkpoint/last.pt 2>/dev/null | head -1)

if [ -z "$LAST_CKPT" ]; then
    echo "No se encontró modelo pre-entrenado. Espera a que termine el pre-entrenamiento."
    exit 1
fi

echo "Evaluando política: $LAST_CKPT"

docker exec  robo_imitate-container bash -c "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0 && export PYTHONPATH=/workspace/ReinFlow:\$PYTHONPATH && export REINFLOW_DIR=/workspace/ReinFlow && export REINFLOW_DATA_DIR=/workspace/ReinFlow/data && export REINFLOW_LOG_DIR=/workspace/ReinFlow/log && export WANDB_MODE=offline && cd /workspace/ReinFlow && python3 script/run.py --config-dir cfg/isaac/eval/xarm_screwdriver --config-name eval_shortcut_mlp base_policy_path=\"$LAST_CKPT\""
