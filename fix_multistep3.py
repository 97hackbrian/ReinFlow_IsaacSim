import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/multi_step.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace:
# observation, reward, done, info = self.env.step(act)
# With:
# step_ret = self.env.step(act)
# if len(step_ret) == 4:
#     observation, reward, done, info = step_ret
# else:
#     observation, reward, terminated, truncated, info = step_ret
#     done = terminated or truncated

content = re.sub(r'observation, reward, done, info = self\.env\.step\(act\)',
"""step_ret = self.env.step(act)
            if len(step_ret) == 4:
                observation, reward, done, info = step_ret
            else:
                observation, reward, terminated, truncated, info = step_ret
                done = terminated or truncated""", content)

# Also check what MultiStep.step returns!
# It currently returns:
# return observation, reward, done, info
content = re.sub(r'return observation, reward, done, info',
"""return observation, reward, terminated, truncated, info""", content)


with open(filepath, 'w') as f:
    f.write(content)
