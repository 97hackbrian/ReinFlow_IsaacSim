#!/bin/bash
set -e

echo "=== 1. Copiando datos y logs desde el contenedor al host ==="
echo "(Te pedirá tu contraseña de sudo para poder copiar archivos protegidos por docker)"
sudo docker cp robo_imitate-container:/workspace/ReinFlow/data/. /home/hackbrian/Documents/gits/ReinFlow_IsaacSim/data/ || true
sudo docker cp robo_imitate-container:/workspace/ReinFlow/log/. /home/hackbrian/Documents/gits/ReinFlow_IsaacSim/log/ || true

echo "=== 2. Otorgando permisos a tu usuario local ==="
sudo chown -R $USER:$USER /home/hackbrian/Documents/gits/ReinFlow_IsaacSim/data/
sudo chown -R $USER:$USER /home/hackbrian/Documents/gits/ReinFlow_IsaacSim/log/

echo "=== 3. Haciendo commit del contenedor actual (reinflow_imitate:latest) ==="
docker commit robo_imitate-container reinflow_imitate:latest

echo "=== 4. Deteniendo contenedor antiguo ==="
docker stop robo_imitate-container || true
# Nota: No lo borramos (rm) por seguridad, solo lo detenemos.

echo "=== 5. Lanzando el nuevo contenedor sincronizado ==="
docker run -it -d \
  --name reinflow_imitate-container \
  --runtime=nvidia \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  --cap-add CAP_SYS_ADMIN \
  --security-opt label=disable \
  --restart unless-stopped \
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /dev:/dev:rw \
  -v /home/hackbrian/.Xauthority:/robo_imitate/.Xauthority:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/dri:/dev/dri:ro \
  -v /home/hackbrian/Documents/gits/robo_imitate:/robo_imitate/ros2_ws/src/robo_imitate \
  -v /home/hackbrian/Documents/gits/robo_imitate/docker/assets:/robo_imitate/assets \
  -v /home/hackbrian/Documents/gits/ReinFlow_IsaacSim:/workspace/ReinFlow \
  reinflow_imitate:latest

echo ""
echo "=== ¡Migración Completada! ==="
echo "Tu nuevo contenedor se llama 'reinflow_imitate-container'."
echo "Ahora /workspace/ReinFlow está 100% sincronizado con tu carpeta local."
