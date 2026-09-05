import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/__init__.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(
    r'device=f"cuda:\{gpu_id\}",',
    r'device=f"cuda:{gpu_id}",\n                normalization_path=normalization_path,',
    content
)

with open(filepath, 'w') as f:
    f.write(content)

