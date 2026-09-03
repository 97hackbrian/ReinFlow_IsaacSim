#!/bin/bash
echo "Iniciando simulacion del entorno Transport (Robomimic)..."
docker exec -it \
    -e PYTHONPATH=. -e PYOPENGL_PLATFORM=osmesa \
    -e REINFLOW_DIR=/workspace/ReinFlow \
    -e REINFLOW_DATA_DIR=/workspace/ReinFlow/data \
    -e REINFLOW_LOG_DIR=/workspace/ReinFlow/log \
    -e DPPO_DATA_DIR=/workspace/ReinFlow/data \
    -e DPPO_LOG_DIR=/workspace/ReinFlow/log \
    -w /workspace/ReinFlow \
    reinflow-mujoco-container bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate reinflow && python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp +denoising_step_list=[4] load_ema=False env.save_video=true +env.render=true env.n_envs=1 render_num=1 device=cpu base_policy_path=/workspace/ReinFlow/log/robomimic/pretrain/transport/transport_pre_shortcut_mlp_ta4_td20/2026-09-03_23-27-47_42/checkpoint/state_1.pt'
echo "Video guardado en visualize/ReFlow/transport/"
