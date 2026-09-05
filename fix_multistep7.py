import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'self\.done = deque\(maxlen=self\.n_action_steps\)\s*self\.cnt = 0',
"""self.done = deque(maxlen=self.n_action_steps)
        self.info = __import__('collections').defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        self.cnt = 0""", content)

with open(filepath, 'w') as f:
    f.write(content)
