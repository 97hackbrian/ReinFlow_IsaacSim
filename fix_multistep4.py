import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'self.obs = deque\(\[obs\], maxlen=max\(self\.n_obs_steps \+ 1, self\.n_action_steps\)\)\s*self\.cnt = 0\s*self\.done = False',
"""self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        self.action = deque(maxlen=self.n_action_steps)
        self.reward = deque(maxlen=self.n_action_steps)
        self.done = deque(maxlen=self.n_action_steps)
        self.cnt = 0""", content)

with open(filepath, 'w') as f:
    f.write(content)
