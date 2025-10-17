# config for training GPT-2 (124M) with 8 communications
# Mix: 6 SPD + 6 Parallel = (6×1) + 2 (boundary) = 8 total

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M-8comm'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 64 #5 * 8

# 20k step experiment (vs 600k for full training)
max_iters = 20000
lr_decay_iters = 20000

# Logging: eval every 250 steps = 80 points over 20k, log train loss every 10 steps
eval_interval = 250
eval_iters = 200
log_interval = 10

weight_decay = 1e-1

# DistributedGPT specific
model_type = 'distributed_gpt'
tp_size = 2
block_types = ['spd', 'parallel'] * 6  # Alternating: 6×(SPD+Parallel) = 6×1 + 2 boundary = 8 comms
compile = False
