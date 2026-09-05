import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/xarm_isaac_env.py'
with open(filepath, 'r') as f:
    content = f.read()

# Add normalization_path to __init__
content = re.sub(
r'random_spawn=False,\n        \*\*kwargs',
r'random_spawn=False,\n        normalization_path=None,\n        **kwargs', content)

# Initialize normalization in __init__
init_code = """        self.random_spawn = random_spawn
        
        self.normalization_path = normalization_path
        self.obs_min, self.obs_max = None, None
        self.action_min, self.action_max = None, None
        if self.normalization_path is not None:
            data = np.load(self.normalization_path)
            self.obs_min = data['obs_min']
            self.obs_max = data['obs_max']
            self.action_min = data['action_min']
            self.action_max = data['action_max']"""

content = re.sub(r'        self\.random_spawn = random_spawn', init_code, content)

# Add normalize and unnormalize methods
methods_code = """    def normalize_obs(self, obs_state):
        if self.obs_min is not None:
            return 2 * ((obs_state - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
        return obs_state
        
    def unnormalize_action(self, action):
        if self.action_min is not None:
            action = (action + 1) / 2.0
            return action * (self.action_max - self.action_min) + self.action_min
        return action
"""
content = re.sub(r'    def step\(self, action\):', methods_code + '\n    def step(self, action):', content)

# Modify step to unnormalize action
content = re.sub(r'        action = np\.clip\(action, -1\.0, 1\.0\)',
r'        action = np.clip(action, -1.0, 1.0)\n        action = self.unnormalize_action(action)', content)

# Modify _get_obs to normalize state
content = re.sub(r'            "state": self\.current_ee_pose\.copy\(\),',
r'            "state": self.normalize_obs(self.current_ee_pose.copy()),', content)

with open(filepath, 'w') as f:
    f.write(content)

