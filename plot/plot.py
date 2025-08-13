from matplotlib import pyplot as plt
import wandb

# what we need to toL
# 1. call wandb api and get the logs
# 2. make it possible to make a scrollable plot, where the scroll is the number of steps, x axis is alpha value (-0.01, -0.005, 0, 0.005), y axis is response_length/mean, draw a line.
# 3. save the plot as a short gif

# 1. load the logs

import wandb
run = wandb.init()
artifact = run.use_artifact('robusteval/ppo_reward_hacking/run-jwh1etj4-history:v2', type='wandb-history')
artifact_dir = artifact.download()

print(artifact_dir)