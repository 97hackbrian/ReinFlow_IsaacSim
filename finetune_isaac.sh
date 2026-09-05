#!/bin/bash
LAST_CKPT=$(docker exec robo_imitate-container bash -c "ls -td /workspace/ReinFlow/log/isaac/pretrain/xarm_screwdriver/*/*/checkpoint/last.pt 2>/dev/null | head -1")

if [ -z "$LAST_CKPT" ]; then
    echo "No se encontró modelo pre-entrenado. Espera a que termine el pre-entrenamiento."
    exit 1
fi

echo "Iniciando fine-tuning con PPO-Flow usando base_policy_path: $LAST_CKPT"

# Trap para asegurar que Ctrl+C mate el proceso dentro del contenedor
trap "echo -e '\nDeteniendo proceso en el contenedor...'; docker exec robo_imitate-container pkill -9 -f run.py; exit 0" SIGINT SIGTERM

docker exec -i robo_imitate-container bash -c "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0 && export PYTHONPATH=/workspace/ReinFlow:\$PYTHONPATH && export REINFLOW_DIR=/workspace/ReinFlow && export REINFLOW_DATA_DIR=/workspace/ReinFlow/data && export REINFLOW_LOG_DIR=/workspace/ReinFlow/log && export WANDB_MODE=offline && export REINFLOW_WANDB_ENTITY=offline && cd /workspace/ReinFlow && python3 script/run.py --config-dir cfg/isaac/finetune/xarm_screwdriver --config-name ft_ppo_reflow_mlp_img base_policy_path=\"$LAST_CKPT\" wandb.offline_mode=True"
