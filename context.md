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

## Plan de Acción Pendiente

1. Ejecutar de forma persistente el `lite6_cartesian_launch.py` dentro de `robo_imitate-container`.
2. Validar que la cadena de comunicación fluya: `ReinFlow` (`/target_frame_raw`) --> `sixd_speed_limiter` (`/target_frame`) --> `cartesian_motion_controller` --> `/isaac/joint_command` --> `Isaac Sim`.
3. Iniciar la fase de **Fine-Tuning con RL (DPPO o PPO-Flow)** usando el pre-entrenamiento base de Flow Matching, ya con el brazo moviéndose libremente en Isaac Sim.
