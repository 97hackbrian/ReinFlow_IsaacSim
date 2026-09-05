import sys

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/xarm_isaac_env.py'
with open(filepath, 'r') as f:
    content = f.read()

content = content.replace("from std_msgs.msg import Float64MultiArray, Twist, Pose", 
                          "from std_msgs.msg import Float64MultiArray\n        from geometry_msgs.msg import Twist, Pose")

with open(filepath, 'w') as f:
    f.write(content)

