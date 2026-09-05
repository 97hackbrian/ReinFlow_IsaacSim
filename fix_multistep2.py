import sys

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

import re
content = re.sub(r'def reset_arg\(self, options_list=None, \*\*kwargs\):.*?return self\._get_obs\(\)', 
"""def reset_arg(self, options_list=None, **kwargs):
        if hasattr(self.env, "reset_arg"):
            obs = self.env.reset_arg(options_list=options_list, **kwargs)
        else:
            obs = self.env.reset()
                
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        self.step_count = 0
        self.done = False
        return self._get_obs()""", content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(content)
