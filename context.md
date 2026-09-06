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

---

## Progreso de Fine-Tuning y Nuevos Bugs Resueltos (Sesión 4-5 Sep 2026)

Durante el inicio y depuración del Fine-Tuning con PPO, se resolvieron problemas críticos de comportamiento físico y aprendizaje:

1. **Bug: Robot no volvía al home físico antes de cada iteración**
   - **Causa:** `xarm_isaac_env.py` enviaba comandos de reset mediante coordenadas cartesianas (`/target_frame_raw`), las cuales eran ignoradas por variaciones extremas. 
   - **Solución:** Se replicó exactamente el mecanismo del script experto original (`robo_imitate`). El entorno `reset()` ahora apaga el `cartesian_motion_controller`, enciende el `joint_trajectory_controller`, publica los ángulos fijos articulares (`[0.00148, 0.06095, 1.164, -0.00033, 1.122, -0.00093]`), espera 3.5s, y vuelve a encender el controlador cartesiano. (✅ Resuelto).

2. **Bug: Ctrl+C en script `.sh` no apagaba el PPO en Docker**
   - **Causa:** Enviar SIGINT desde el host al proceso de `docker exec` no mataba el subproceso de `python script/run.py` dentro del contenedor.
   - **Solución:** Se incluyó un bloque de captura (trap) en `finetune_isaac.sh`: `trap "docker exec robo_imitate-container pkill -f script/run.py; exit 0" SIGINT SIGTERM`. Ahora finaliza el script remotamente sin dejar procesos zombie. (✅ Resuelto).

3. **Bug Crítico de RL: PPO explotaba en `NaN` o la política se congelaba ("se queda quieto")**
   - **Causa 1 (PPO Congelado):** Sin multiplicador de velocidad, el modelo pre-entrenado daba pasos físicos de *1.5 milímetros*. El sistema RL original usaba recompensa densa exponencial que caía a `0.0` a distancias largas. Como el robot avanzaba lentísimo y no ganaba recompensa, PPO aprendía rápidamente a no moverse para minimizar el `r_penalty` de acción.
   - **Causa 2 (Explosión NaN):** Al agregar de vuelta el factor empírico de `speed_multiplier = 40.0` (usado originalmente para inferencia visual), la exploración aleatoria natural de RL (`std=0.1`) provocó que se predijeran acciones máximas (`0.264 m`), que al multiplicarse por 40 resultaban en comandos de **10 metros por paso**, rompiendo el motor de física de Isaac Sim y resultando en poses `NaN` para el Actor/Crítico de PyTorch.
   - **Solución Híbrida Aplicada:** 
     1. Se **eliminó permanentemente** el `speed_multiplier=40.0` de `xarm_isaac_env.py`.
     2. Se ajustó `r_reach` en la función de recompensa hacia un modelo denso completamente lineal (`r_reach = -dist_xy * 5.0`). Esto otorga un gradiente fuerte en cada paso (mientras más cerca, menos penalización), incitando a PPO a **aprender por su propia cuenta** a aumentar la velocidad (tamaño de acción) de manera segura y controlada sin necesidad de un parche multiplicador. (✅ Resuelto, iteraciones iterativas estables validadas en log sin bloqueos).

| Aspecto | Esperado/Observado Actualizado (5 Sep) |
|---------|--------------------------------|
| Movimiento EEF | El robot realiza reset físico articular exitosamente en iteración 0 y navega el espacio hacia el objetivo. |
| Valores `/target_frame_raw` | Coordenadas absolutas enviadas correctamente y respetando el rango físico. |
| Episode Reward | Comenzó en `0.00` en iteración 0 y 1 (hasta llenarse el max_episode_steps de 400). Valores Q captaron señal positiva de recompensa densa. |


### 5. Finalización del Pre-entrenamiento (5 Sep 2026)
- Se ejecutó el pre-entrenamiento completo (200 épocas) exitosamente (`task-2433`).
- El loss convergió de manera óptima a `0.0096`, superando con creces la meta de `0.02`. 

### 6. Bug de Normalización de Datos y Vuelo del Robot
- **Síntoma:** Tras el pre-entrenamiento, durante la evaluación el robot logró acercarse al destornillador en X e Y, pero en lugar de bajar a agarrarlo, salió volando hacia arriba (eje Z positivo).
- **Análisis de Causa Raíz:** Se revisó el pipeline de `robo_imitate_to_reinflow.py` y se descubrió que el dataset teleoperado contiene un outlier o "salto" masivo de **-26.4 centímetros** (`-0.264m`) en el primer paso de todos los episodios (probablemente el "snap" de inicio del SpaceMouse).
- **Efecto de Normalización:** Como los datos se aplastan al rango `[-1, 1]`, este outlier comprimió los movimientos sutiles y útiles del agarre (que rondan los `2 milímetros`) hacia el extremo `0.99` de la red. En consecuencia, si la red neuronal predice `0.0` (duda o inmovilidad), al des-normalizarse se convierte en un violento salto de **-13 centímetros**.
- **Soluciones Intentadas y Crasheo (Eigenvalues did not converge):**
  - El usuario invirtió la polaridad manualmente haciendo `act_mat[2, 3] = -act_z`. Esto corrigió la dirección (el robot empezó a ir hacia abajo).
  - Sin embargo, como el salto seguía siendo de 13 centímetros, el robot se estrelló contra la mesa a velocidad terminal. 
  - El choque provocó que el motor de físicas de Isaac Sim colapsara y entregara coordenadas inválidas (`NaN`), lo que desencadenó en el error matemático `numpy.linalg.LinAlgError: Eigenvalues did not converge` dentro de la librería `transforms3d` al intentar calcular los cuaterniones.
  
### 7. Ajustes en Reinforcement Learning (PPO)
- Se corrigió el archivo `ft_ppo_reflow_mlp_img.yaml`:
  - Se aumentó `n_steps: 1000` y `max_episode_steps: 1000` para que cada iteración de PPO abarque una trayectoria completa.
  - Se añadió `reset_at_iteration: false` para evitar que el entorno aborte y reinicie las trayectorias prematuramente a la mitad de una aproximación.

### 8. Próximos Pasos
- El usuario ha solicitado pausar y revisar a profundidad el proceso de Flow Matching antes de continuar, a la espera de nuevas instrucciones.

### 9. Depuración de Evaluación de Flow Matching (5 Sep 2026 - Noche)
Tras resolver el eje Z, el usuario reportó que el script de evaluación (`eval_isaac.sh`) interrumpía el robot a medio camino y lo regresaba a su posición inicial, fallando la tarea. Se identificaron y solucionaron múltiples bugs que causaban este "reseteo fantasma":

1. **Bug de Tiempo Físico vs. Tiempo de Inferencia:** 
   - **Causa:** El archivo YAML especificaba `max_episode_steps: 1000` (pensando en 1000 pasos de red neuronal). Sin embargo, el wrapper intermedio (`multi_step.py`) lo interpretaba como pasos físicos del simulador. Como cada predicción equivale a 4 pasos físicos, el wrapper cortaba la simulación al llegar al paso de red neuronal 250 (250 * 4 = 1000), abortando prematuramente el episodio.
   - **Solución:** Se editó `multi_step.py` para multiplicar automáticamente `max_episode_steps * n_action_steps`, permitiendo a la simulación vivir los 4000 pasos físicos correspondientes a los 1000 pasos de inferencia.

2. **Bug de Bloqueo de la Animación de Agarre (Timeout):**
   - **Causa:** Al detectar el objetivo (Z < 0.18m), el código de Python enviaba la orden de bajar y se "dormía" por 3 segundos (`time.sleep(3.0)`). Pero el nodo de control de ROS (`sixd_speed_limiter`) exige recibir mensajes constantemente; al pasar 0.5 segundos de silencio, congelaba el brazo por seguridad.
   - **Solución:** Se reemplazó el `time.sleep` estático por un bucle `while` que envía continuamente la posición deseada a ROS en iteraciones de 0.1s.

3. **Bug del Límite de Velocidad (Falta de recorrido en Z):**
   - **Causa:** El usuario notó que al brazo "le faltaban 10 cm para llegar abajo" durante el agarre. Se descubrió que `sixd_speed_limiter` restringe la velocidad máxima a 2 centímetros por segundo (0.02 m/s). Para bajar 16 cm (de 0.22 a 0.06), el robot requería 8 segundos físicos, pero el código anterior solo le daba 3 segundos (logrando recorrer solo 6 cm).
   - **Solución:** Se eliminó el tiempo fijo y se implementó un sistema de espera dinámica. Ahora el programa escanea la coordenada Z actual del efector final (`self.current_ee_pose[2]`) y espera inteligentemente el tiempo que sea necesario (hasta un límite de 15s) hasta que el brazo alcance físicamente la cota de agarre (`0.06m`).

Además, se corrigió un `AttributeError: save_video` inicializando correctamente la variable desde el yaml en `eval_agent_base.py`, y se configuró este último para abortar limpiamente el bucle y cerrar el programa en cuanto la secuencia de agarre declara el `terminated=True`.
