#!/bin/bash
# Script para lanzar la visualización en vivo del manipulador robótico (Kitchen)
# Asegúrate de haber ejecutado 'xhost +local:docker' en tu máquina host antes de correr esto.

echo "Iniciando la simulación visual del manipulador robótico entrenado..."
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
             --config-dir=cfg/gym/eval/kitchen-complete-v0 \
             --config-name=eval_reflow_mlp \
             "denoising_step_list=[4]" \
             +load_ema=False \
             env.save_video=false \
             +env.render=true \
             env.n_envs=1 \
             device=cuda:0 \
             sim_device=cuda:0'
