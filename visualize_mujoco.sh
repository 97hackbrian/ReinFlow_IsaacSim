#!/bin/bash
# Script para lanzar la visualización en vivo del robot en MuJoCo
# Asegúrate de haber ejecutado 'xhost +local:docker' en tu máquina host antes de correr esto.

echo "Iniciando la simulación visual del modelo entrenado..."
echo "Deberías ver una ventana abriéndose en breve."

docker exec -it \
  -e PYTHONPATH=. \
  -e REINFLOW_DIR=/workspace/ReinFlow \
  -e REINFLOW_DATA_DIR=/workspace/ReinFlow/data \
  -e REINFLOW_LOG_DIR=/workspace/ReinFlow/log \
  -w /workspace/ReinFlow \
  reinflow-mujoco-container \
  bash -c 'source /opt/conda/etc/profile.d/conda.sh && \
           conda activate reinflow && \
           python script/run.py \
             --config-dir=cfg/gym/eval/walker2d-medium-v2 \
             --config-name=eval_reflow_mlp \
             base_policy_path=/workspace/ReinFlow/log/gym/finetune/walker2d-medium-v2_ppo_reflow_mlp_ta4_td4_tdf4/2026-09-03_21-11-40_seed42/checkpoint/best.pt \
             "denoising_step_list=[1,2]" \
             load_ema=False \
             env.save_video=false \
             +env.render=true \
             env.n_envs=1 \
             device=cuda:0 \
             sim_device=cuda:0'
