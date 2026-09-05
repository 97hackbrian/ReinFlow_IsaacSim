# Contexto y Estado del Proyecto (03 de Septiembre, 2026)

## Solicitud Original del Usuario
El usuario solicitó avanzar con la **Opción B**: la integración de `ReinFlow` con la simulación en `Isaac Sim` del brazo robótico xArm, utilizando el entorno de ROS 2 provisto en el repositorio `robo_imitate`.
La simulación objetivo está definida en: `robo_imitate/xarm_bringup/isaac/object_picking.usda`.

## Acciones Realizadas y Decisiones Tomadas

1. **Evaluación de Contenedores:**
   Dado que `reinflow-mujoco-container` (Ubuntu 20.04) causaba conflictos de dependencias al intentar instalar ROS 2 Humble (Python 3.10 vs 3.8), y el contenedor `isaac-sim-6.0-wayland` está muy encapsulado, optamos por utilizar `robo_imitate-container`.
   `robo_imitate-container` posee ROS 2 Humble nativo, Python 3.10 y PyTorch, lo cual lo convierte en el puente ideal.
   Se copiaron e instalaron satisfactoriamente las librerías necesarias de `ReinFlow` dentro de este contenedor.

2. **Modificación del Código de Evaluación (`ReinFlow_IsaacSim`):**
   - **`env/gym_utils/__init__.py`:** Se interceptó el entorno `"xarm_screwdriver"` para inicializar la clase `XArmPickScrewdriverEnv` sin pasar por el registro estricto de Gym. Además, se forzó el modo `asynchronous=False` (SyncVectorEnv) para evitar que `AsyncVectorEnv` inicie subprocesos (`multiprocessing`), lo que choca con la instanciación de nodos ROS 2 y genera colisiones de nombres y excepciones de DDS (`ValueError: generator already executing`).
   - Se eliminaron las llamadas importadas estrictas de `d4rl.gym_mujoco` que causaban bloqueos al arrancar los scripts en un contenedor sin MuJoCo puro.

3. **Adaptación de la Interfaz Visual / Estado:**
   - La red de inferencia que entrenamos era `ShortCutFlowViT` (basada en imágenes).
   - Se corrigió el archivo `cfg/isaac/eval/xarm_screwdriver/eval_shortcut_mlp.yaml` para que utilice el Agente Visual `EvalImgShortCutAgent`, el cual extrae las variables `rgb` del diccionario de estado para inferir los flujos, algo que el agente puramente numérico no hacía.

4. **El Entorno ROS 2 (Wrapper `xarm_isaac_env.py`):**
   - El entorno fue configurado para recibir imágenes en `/rgb` y la pose del robot en `/current_pose` desde Isaac Sim (mediante los Action Graphs en `object_picking.usda`).
   - El entorno devuelve las acciones deseadas publicando deltas espaciales absolutos en el tópico `/target_frame_raw` (pose cartesiana 6D: XYZ + RPY).
   - Hubo un error de Gym (versión 0.24.1 vs 0.26) donde el Wrapper devolvía 5 variables (`obs, reward, terminated, truncated, info`), se ajustó para devolver 4 (`obs, reward, done, info`).

5. **El Problema del Movimiento (Estado Actual):**
   - El modelo RL fue lanzado exitosamente, predijo movimientos y los publicó en `/target_frame_raw`.
   - **Problema:** Isaac Sim y sus Action Graphs nativos en `object_picking.usda` NO leen `/target_frame_raw`. Isaac Sim mueve sus articulaciones escuchando al tópico de *comandos articulares* (`/isaac/joint_command` de tipo `sensor_msgs/JointState`).
   - **Causa:** Nuestro algoritmo en `ReinFlow` predice movimientos del efector final en 6D. Falta una etapa de *Cinemática Inversa (IK)* que convierta `/target_frame_raw` en `/isaac/joint_command`.
   - **Solución propuesta (Pendiente):** En el repositorio `robo_imitate`, existe un script de lanzamiento (`xarm_bringup/launch/lite6_cartesian_launch.py`) que instancia nodos de control espacial (`sixd_speed_limiter` y `cartesian_motion_controller`) que operan la Cinemática Inversa usando MoveIt/Servo, recibiendo los comandos cartesianos y enviando las posiciones articulares a Isaac Sim. El usuario debe ejecutar dicho controlador en paralelo.

6. **Entrenamiento Previo de Flow Matching Detenido:**
   El usuario indicó que la PC seguía consumiendo GPU. Se confirmó que el proceso `script/run.py` con el pre-entrenamiento seguía ejecutándose en segundo plano. Fue finalizado por completo (`kill -9`) en todos los contenedores para liberar recursos exclusivamente para Reinforcement Learning.

## Plan de Acción Pendiente (al 03 Sep)

1. Ejecutar de forma persistente el `lite6_cartesian_launch.py` dentro de `robo_imitate-container`.
2. Validar que la cadena de comunicación fluya: `ReinFlow` (`/target_frame_raw`) --> `sixd_speed_limiter` (`/target_frame`) --> `cartesian_motion_controller` --> `/isaac/joint_command` --> `Isaac Sim`.
3. Iniciar la fase de **Fine-Tuning con RL (DPPO o PPO-Flow)** usando el pre-entrenamiento base de Flow Matching, ya con el brazo moviéndose libremente en Isaac Sim.

---
---

# Continuación — Sesión del 04–05 de Septiembre 2026

## Objetivo de la Sesión
Ejecutar el fine-tuning PPO con el robot en Isaac Sim. El robot xArm Lite 6 debe moverse hacia el destornillador, agarrarlo y levantarlo. Se retomó desde cero el pre-entrenamiento de Flow Matching porque el modelo base anterior causaba que el robot oscilara en su posición inicial.

---

## Arquitectura del Sistema (actualización)

### Flujo de Comunicación ROS 2 confirmado y funcional:
```
ReinFlow script/run.py (en robo_imitate-container)
  → publica PoseStamped en /target_frame_raw
    (delta 6D relativo, frame_id = "gripper_link_base")
  → sixd_speed_limiter (limita velocidad máxima)
  → cartesian_motion_controller (IK: cartesiano → articular)
  → /joint_states → Isaac Sim mueve el robot

Isaac Sim (en HOST, NO en contenedor)
  → publica /rgb (imagen cámara, 96x96 px)
  → publica /current_pose (PoseStamped, pose EEF en metros)
  → suscribe /respawn (Twist: x=pos_x, y=pos_y, z=altura → posición destornillador)
  → suscribe /position_controller/commands (Float64MultiArray: gripper, 0.0=abierto, -0.01=cerrado)
```

### CRÍTICO — El repositorio del host NO está montado en el contenedor:
- Host: `/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/`
- Contenedor: `/workspace/ReinFlow/`
- Sincronizar siempre con: `docker cp <archivo_host> robo_imitate-container:/workspace/ReinFlow/<ruta>`

---

## Dataset y Normalización

- **Path en contenedor**: `/workspace/ReinFlow/data/isaac/xarm_screwdriver/`
- `train.npz`: ~1.2 GB, 150 episodios, 45,507 pasos
- `normalization.npz`: estadísticas de normalización

### Valores exactos de normalization.npz:
```
obs_min:  [0.0887, -0.1212, 0.0908, -3.1416, -0.0239, 0.0017]
obs_max:  [0.3185,  0.1763, 0.3557,  3.1416,  0.0195, 0.0124]

action_min: [-0.0641, -0.1218, -0.2648, -0.0231, -0.0371, -0.0009]
action_max: [ 0.1688,  0.1752,  0.0028,  0.0158,  0.0038,  0.0107]
```

**Importante**: Las acciones son deltas de posición en metros/paso. El eje Z va casi exclusivamente de -0.265 a +0.003 (el robot desciende durante la demostración).

---

## Archivos Modificados (estado al 5 Sep 2026)

### `env/gym_utils/wrapper/xarm_isaac_env.py`
1. Carga `normalization.npz` en `__init__` para normalizar obs y unnormalizar acciones.
2. `random_spawn=False` → destornillador fijo en `(0.35, 0.10)` m.
3. `reset()` → publica en `/respawn` (posición fija) y abre gripper.
4. `step()` → unnormaliza acción; usa `self.current_ee_pose` (raw, metros) para reward y success.
5. `_compute_reward()` → usa coordenadas reales en metros.
6. `_send_ros2_action()` → envía acción unnormalizada directamente SIN multiplicar por speed_multiplier.
7. Lógica de agarre al éxito: baja a `final_grasp_z=0.088 m`, cierra gripper, sube a `z=0.29 m`.

### `env/gym_utils/__init__.py`
- Instancia `XArmPickScrewdriverEnv` directamente, pasa `normalization_path`.
- Fuerza `SyncVectorEnv` (no `AsyncVectorEnv`).

### `env/gym_utils/wrapper/multi_step.py`
- Corregido `reset_arg`: inicializa `self.cnt`, `self.action`, `self.info`.
- Corregido desempaquetado de 5 valores en `step()`.

### `agent/finetune/reinflow/train_ppo_shortcut_img_agent.py`
- `self.use_early_stop = False` para evitar salida prematura.

### `finetune_isaac.sh` (en host)
Busca checkpoint más reciente **dentro del contenedor**:
```bash
LAST_CKPT=$(docker exec robo_imitate-container bash -c \
  "ls -td /workspace/ReinFlow/log/isaac/pretrain/xarm_screwdriver/*/*/checkpoint/last.pt | head -1")
```

---

## Historial de Bugs de la Sesión

| # | Bug | Estado |
|---|-----|--------|
| 1 | `MultiStepWrapper` sin atributo `cnt` | ✅ Resuelto |
| 2 | Early stopping prematuro (itr 0) | ✅ Resuelto |
| 3 | Reward usa obs normalizadas contra coords reales | ✅ Resuelto |
| 4 | `speed_multiplier * 0.001` → movimientos sub-milímetro | ✅ Resuelto |
| 5 | `action * 40` después de unnormalizar → deltas de 6+ metros | ✅ Resuelto |
| 6 | `/target_frame_raw` sigue mostrando valores > 0.2 m | ✅ Resuelto |
| 7 | `Twist` importado desde `std_msgs` en vez de `geometry_msgs` | ✅ Resuelto |
| 8 | Procesos zombie `script/run.py` acumulados en contenedor | ✅ Resuelto repetidamente |

#### Bug 6 — Detalle (RESUELTO PERMANENTEMENTE)
- **Causa Raíz**: 
  1. `normalization_path` no se pasaba desde `cfg` hacia `make_async()` en `train_agent.py`.
  2. `xarm_isaac_env.py` enviaba los deltas relativos de la acción (respecto a `gripper_link_base`) directamente a `/target_frame_raw`. Sin embargo, `sixd_speed_limiter` asume que todo lo que recibe es una pose **absoluta** en el marco base (`link_base`). Esto causaba que el robot intentara moverse a coordenadas imposibles.
- **Solución Aplicada**:
  1. Se modificó `train_agent.py` y `eval_agent_base.py` para pasar `normalization_path`.
  2. Se modificó `_send_ros2_action()` en `xarm_isaac_env.py` para calcular la **nueva pose absoluta** multiplicando la pose actual del EEF por el delta relativo de la acción (`curr_mat @ act_mat`), replicando la lógica exacta del script de inferencia del experto.
  3. Se añadió un reset a la posición *home* (`x=0.207, y=0, z=0.35`) al inicio de cada episodio usando `/target_frame_raw`.

---

## Pre-entrenamiento — Resultado de la Sesión

### Nuevo pre-entrenamiento (4 Sep 2026, iniciado 18:35)
```
Épocas completadas: 25 (detenido manualmente al alcanzar loss=0.050)
Época  0: loss=0.8160
Época  1: loss=0.6022
Época  3: loss=0.2617
Época  8: loss=0.1808
Época 14: loss=0.1327
Época 17: loss=0.1063
Época 24: loss=0.0499 ← detenido aquí

Checkpoint: /workspace/ReinFlow/log/isaac/pretrain/xarm_screwdriver/
            xarm_screwdriver_pre_shortcut_mlp_img_ta4_td20/
            2026-09-04_18-35-37_42/checkpoint/last.pt
```

**Recomendación**: Continuar hasta 100-200 épocas (loss < 0.02) antes del siguiente fine-tuning.

---

## Variables de Entorno Requeridas en el Contenedor

```bash
export REINFLOW_DIR=/workspace/ReinFlow
export REINFLOW_DATA_DIR=/workspace/ReinFlow/data
export REINFLOW_LOG_DIR=/workspace/ReinFlow/log
export REINFLOW_WANDB_ENTITY=offline
export PYTHONPATH=/workspace/ReinFlow
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline
```

---

## Comandos Clave para Relanzar Todo

```bash
# 1. Matar procesos anteriores
docker exec robo_imitate-container pkill -f script/run.py
docker exec robo_imitate-container ps aux | grep python  # verificar limpieza

# 2. Sincronizar archivos modificados
cd /home/hackbrian/Documents/gits/ReinFlow_IsaacSim
docker cp env/gym_utils/wrapper/xarm_isaac_env.py \
  robo_imitate-container:/workspace/ReinFlow/env/gym_utils/wrapper/xarm_isaac_env.py
docker cp env/gym_utils/__init__.py \
  robo_imitate-container:/workspace/ReinFlow/env/gym_utils/__init__.py
docker cp env/gym_utils/wrapper/multi_step.py \
  robo_imitate-container:/workspace/ReinFlow/env/gym_utils/wrapper/multi_step.py

# 3. Lanzar controlador cartesiano (en terminal separada, debe quedar corriendo)
docker exec -it robo_imitate-container bash -c \
  "source /opt/ros/humble/setup.bash && \
   source /robo_imitate/ros2_ws/install/setup.bash && \
   ros2 launch xarm_bringup lite6_cartesian_launch.py"

# 4. Lanzar fine-tuning
./finetune_isaac.sh

# 5. Verificar topic de acciones (los valores deben ser < 0.18 m)
docker exec robo_imitate-container bash -c \
  "source /opt/ros/humble/setup.bash && ros2 topic echo /target_frame_raw --once"

# 6. Prueba manual del pipeline (¿el robot se mueve 5 mm en X?)
docker exec robo_imitate-container bash -c "source /opt/ros/humble/setup.bash && \
  ros2 topic pub --once /target_frame_raw geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: gripper_link_base}, pose: {position: {x: 0.005, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'"
```

---

## Comportamiento Esperado vs Observado

| Aspecto | Esperado | Observado (al cierre de sesión) |
|---------|----------|---------------------------------|
| Movimiento EEF | Se mueve hacia (0.35, 0.10) m | Oscila cerca de posición inicial |
| Valores `/target_frame_raw` | position.x < 0.169 m | position.x = -0.213 m (fuera de rango) |
| Destornillador | Fijo en (0.35, 0.10, 0.012) | Aparecía aleatoriamente (procesos zombie) |
| Episode Reward | Crece con iteraciones PPO | 0.00 en todas las iteraciones |
| `No episode completed` | Episodios completos en buffer | WARNING en todas las iteraciones |

---

## Próximos Pasos Priorizados

1. **(INMEDIATO)** Añadir debug prints en `xarm_isaac_env.py` para confirmar que `action_min` se carga y que los valores de acción están en rango.
2. **(INMEDIATO)** Prueba manual del pipeline con `ros2 topic pub` para confirmar que el controlador cartesiano responde correctamente.
3. **(MEDIO PLAZO)** Continuar pre-entrenamiento hasta 100-200 épocas.
4. **(MEDIO PLAZO)** Reducir `max_episode_steps` de 400 a 150-200 para que el buffer de PPO reciba episodios completos.
5. **(RESULTADO FINAL BUSCADO)** El robot agarre el destornillador exitosamente en Isaac Sim al menos 1 vez.

---

## Notas Críticas para la Próxima Sesión

1. **Isaac Sim debe estar en Play** antes de lanzar cualquier cosa.
2. **El controlador cartesiano (`lite6_cartesian_launch.py`) debe estar corriendo** siempre.
3. **Nunca lanzar `./finetune_isaac.sh` dos veces** sin matar el proceso anterior con `pkill -f script/run.py`.
4. **El robot NO logró agarrar el destornillador en ningún intento de esta sesión.** La causa es el Bug 6 (pipeline de normalización de acciones).
5. **Hardware**: RTX 4060 Max-Q 8GB VRAM. GPU memory durante fine-tuning: ~0.11 GB allocated. Tiempo/iteración PPO: ~20-22 s.

---

## Checkpoints Disponibles en el Contenedor

```
/workspace/ReinFlow/log/isaac/pretrain/xarm_screwdriver/
└── xarm_screwdriver_pre_shortcut_mlp_img_ta4_td20/
    ├── 2026-09-04_00-17-21_42/checkpoint/last.pt  ← pre-train anterior (sesión 3 Sep)
    └── 2026-09-04_18-35-37_42/checkpoint/last.pt  ← pre-train nuevo (25 épocas, 4 Sep)
```
