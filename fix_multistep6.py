import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'observation, reward, terminated, truncated, info = step_ret\s*done = terminated or truncated',
"""observation, reward, terminated, truncated, info = step_ret
                if isinstance(info, (list, tuple)) and len(info) > 0: info = info[0]
                done = terminated or truncated""", content)
                
content = re.sub(r'observation, reward, done, info = step_ret',
"""observation, reward, done, info = step_ret
                if isinstance(info, (list, tuple)) and len(info) > 0: info = info[0]""", content)

with open(filepath, 'w') as f:
    f.write(content)
