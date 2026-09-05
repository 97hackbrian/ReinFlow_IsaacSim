import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/agent/finetune/reinflow/train_ppo_shortcut_img_agent.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'self\.use_early_stop = True', 'self.use_early_stop = False', content)

with open(filepath, 'w') as f:
    f.write(content)

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/agent/finetune/reinflow/train_ppo_flow_img_agent.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'self\.use_early_stop = True', 'self.use_early_stop = False', content)

with open(filepath, 'w') as f:
    f.write(content)
