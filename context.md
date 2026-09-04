# Contexto de la Sesión y Plan de Integración: ReinFlow + Isaac Sim

Este documento sirve como registro detallado del progreso, conocimiento y configuraciones (hiperparámetros) descubiertos a lo largo de nuestras sesiones de depuración. Además, establece una hoja de ruta técnica clara para integrar la simulación de Isaac Sim 6.0 y el manipulador robótico xArm con el algoritmo de ReinFlow.

## 1. Contexto y Problemas Solucionados

Durante esta fase, operamos bajo la **"Opción A"** (validación de los baselines de MuJoCo y Robosuite dentro de un contenedor Docker en Ubuntu 24.04 con Wayland/Hyprland) para asegurar que el repositorio de `ReinFlow` fuera totalmente funcional antes de realizar modificaciones profundas para Isaac Sim.

### 1.1 Corrección del Pipeline de Visualización de MuJoCo / Robosuite
- **Renderizado Offscreen (OSMesa):** La evaluación en Robomimic fallaba al no contar con un servidor EGL disponible. Forzamos la variable `PYOPENGL_PLATFORM=osmesa` y desactivamos `render_onscreen` en los scripts de evaluación para permitir el uso de `mode="rgb_array"`.
- **VectorEnv Bug de Render:** `SyncVectorEnv` en la implementación de Gym no propagaba correctamente la llamada `render()`. Modificamos `eval_agent_base.py` para llamar explícitamente a `self.venv.envs[i].render(mode='rgb_array')`, redimensionando los cuadros resultantes vía OpenCV (`cv2.resize()`) a 640x480 antes de compilarlos con `cv2.VideoWriter`.

### 1.2 Resolución de Discordancias de Dimensiones en el Espacio de Estados
- **Entorno `can` (19 vs 23 dimensiones):** El dataset de `lift` tenía objetos descritos por un tensor de tamaño 10 (total 19 dims), pero el script usaba el entorno `PickPlaceCan` cuyo tamaño era 14 (total 23 dims). Se corrigió reescribiendo `cfg/robomimic/env_meta/can.json` para que el `env_name` usara `Lift`, empatando a 19.
- **Entorno `transport` (50 vs 59 dimensiones):** Faltaban las variables de estado cinemático del segundo brazo (`robot1`). Se agregó `['robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos']` a la lista de `low_dim_keys` en `eval_shortcut_mlp.yaml` y `pre_shortcut_mlp.yaml` para alinear las 59 dimensiones.
- **Embeddings del Modelo Flow:** Como el campo vectorial requiere una dimensión par para el codificador sinusoidal de posición, se corrigió usando una proyección lineal auxiliar en la red neuronal configurando `td_emb_dim: 128` y `cond_mlp_dims: [128]`.

---

## 2. Configuración e Hiperparámetros del Modelo

Para recrear el nivel de rendimiento óptimo visto en los GIFs originales, el pre-entrenamiento y posterior afinamiento (RL) deben gobernarse bajo configuraciones rigurosas.

### 2.1 Pre-entrenamiento (Imitation Learning vía Flow Matching)
Para alcanzar convergencia completa:
- **`n_epochs: 3000`**: La cantidad requerida de épocas para que el modelo mapee correctamente el ruido gaussiano a distribuciones de acciones complejas.
- **`first_cycle_steps: 3000`**: Programador de tasa de aprendizaje (Learning Rate Scheduler) con un perfil de coseno, garantizando descenso de gradiente hasta la última época.
- **`denoising_steps: 20`**: Resolución de la ODE durante el entrenamiento.
- **`eval_denoising_steps` o `denoising_step_list: [1, 4]`**: Pasos ultra reducidos durante inferencia/evaluación, habilidad nativa de **ShortCutFlow** que garantiza simulaciones a más de 170 HZ sin perder precisión.
- **`horizon_steps: 4`**: La red predice 4 acciones futuras secuenciales en un solo paso (Action Chunking) para mitigar errores acumulativos.
- **`cond_steps: 1`**: Utiliza un 1 paso de estado previo como contexto del mundo.

### 2.2 Fine-Tuning Interactivo con PPO (D-PPO)
En la fase de Reinforcement Learning interactivo (archivo de referencia `ft_ppo_diffusion_mlp.yaml`):
- **`clip_ploss_coef: 0.01`**: A diferencia del PPO estándar (0.2), el recorte aquí es minúsculo para no romper el campo vectorial delicado pre-entrenado de la fase de Flujo/Difusión.
- **`update_epochs: 10`**: Cantidad de épocas PPO por cada tanda de experiencia recolectada.
- **`batch_size: 7500` a `15000`**: Tamaño de experiencia recolectada (e.g. 50 entornos paralelos corriendo por 300 pasos).
- **`target_kl: 1.0`**: Límite de KL divergence que frena prematuramente las actualizaciones para conservar confianza en la política original.
- **`vf_coef: 0.5`**: Coeficiente clásico de pérdida de la función de Valor (Crítico).

---

## 3. Plan Detallado: Integración con Isaac Sim 6.0 y Robot xArm

Habiendo dominado la arquitectura del código, la **"Opción B"** implica migrar las capacidades probadas al simulador fotorrealista de NVIDIA y al brazo robótico xArm.

### Fase 1: Extracción y Transformación de Datos (Data Pipeline)
1. Extraer los datos grabados de teleoperación del repositorio `robo_imitate` (archivos `sim_env_data.parquet`).
2. Adaptar el script local `data_process/robo_imitate_to_reinflow.py` para mapear las imágenes RGB, acciones del xArm Lite 6 (coordenadas articulares/EEF), y transformarlo al formato estructurado `.npz` y archivo `normalization.npz` que utiliza ReinFlow (separando validación y entrenamiento).

### Fase 2: Implementación del Entorno (Gym Wrapper)
1. Crear el wrapper de entorno **`env/gym_utils/wrapper/xarm_isaac_env.py`** que herede de `gym.Env`.
2. Este wrapper debe conectarse de forma nativa u offscreen a **Isaac Sim 6.0** usando Omniverse ISAAC Gym/RL Framework.
3. Cargar la escena principal a través de los archivos `.usda` (`object_picking.usda`) suministrados desde `robo_imitate`.
4. El método `step(action)` deberá inyectar comandos de posición del End-Effector u orientaciones al xArm, correr los pasos de física en Isaac Sim, y extraer la imagen de la cámara y estado a través de la API `get_rgb()` implementada en el wrapper (retornando arrays de numpy).
5. **Recompensa:** Escribir una función de recompensa densa o *sparse* (según la distancia de la pinza al objeto) para que la fase RL posea retroalimentación.

### Fase 3: Pipeline de Entrenamiento unificado
1. Crear un archivo `cfg/isaac/pretrain/pre_shortcut_mlp.yaml` apuntando al dataset extraído del xArm.
2. Entrenar usando `agent.pretrain.train_shortcut_agent.TrainShortCutAgent` para pre-entrenar el campo vectorial del comportamiento teleoperado del xArm recogiendo la caja/tornillo en la escena visual (Imitation Learning).
3. Evaluar el desempeño preliminar usando una versión modificada de nuestro script actual de visualización, guardando videos (`.mp4`) del robot xArm.
4. Pasar a la etapa final de RL, creando `cfg/isaac/finetune/ft_ppo_shortcut_mlp.yaml` que invoca `agent.finetune.train_ppo_diffusion_agent` conectándose al Wrapper interactivo de Isaac Sim 6.0 para pulir y asegurar una tasa de éxito máxima del Pick & Place.
