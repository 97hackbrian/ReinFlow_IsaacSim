import sys

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

# Remove the bad patch at the end
clean_lines = []
for line in lines:
    if line.startswith("    def reset_arg(self, options_list=None, **kwargs):"):
        break
    clean_lines.append(line)

# Find if __name__ == "__main__":
insert_idx = len(clean_lines)
for i, line in enumerate(clean_lines):
    if line.strip().startswith('if __name__ == "__main__":'):
        insert_idx = i
        break

patch = """
    def reset_arg(self, options_list=None, **kwargs):
        if hasattr(self.env, "reset_arg"):
            obs = self.env.reset_arg(options_list=options_list, **kwargs)
        else:
            # Fallback for SyncVectorEnv that doesn't have reset_arg patched
            obs = []
            for i, e in enumerate(self.env.envs):
                o = e.reset(options=options_list[i] if options_list else None)
                if isinstance(o, tuple): o = o[0]
                obs.append(o)
            import numpy as np
            if isinstance(obs[0], np.ndarray):
                obs = np.stack(obs)
                
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        self.step_count = 0
        self.done = False
        return self._get_obs()

"""
clean_lines.insert(insert_idx, patch)

with open(filepath, 'w') as f:
    f.writelines(clean_lines)
