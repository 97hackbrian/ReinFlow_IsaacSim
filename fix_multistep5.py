import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'def _add_info\(self, info\):\s*for key, value in info\.items\(\):',
"""def _add_info(self, info):
        if isinstance(info, (list, tuple)):
            if len(info) > 0 and isinstance(info[0], dict):
                info = info[0] # Just take the first one since n_envs=1 for Isaac
            else:
                info = {}
        for key, value in info.items():""", content)

with open(filepath, 'w') as f:
    f.write(content)
