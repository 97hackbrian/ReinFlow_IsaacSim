import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'return np\.max\(data\)', 'return np.max(data, axis=0)', content)
content = re.sub(r'return np\.min\(data\)', 'return np.min(data, axis=0)', content)
content = re.sub(r'return np\.mean\(data\)', 'return np.mean(data, axis=0)', content)
content = re.sub(r'return np\.sum\(data\)', 'return np.sum(data, axis=0)', content)

with open(filepath, 'w') as f:
    f.write(content)
