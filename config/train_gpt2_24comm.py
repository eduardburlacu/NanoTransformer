# config for training GPT-2 (124M) with 24 communications (baseline)
# All TP blocks: 2 comms/layer × 12 layers = 24 total

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M-24comm'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# 20k step experiment (vs 600k for full training)
max_iters = 20000
lr_decay_iters = 20000

# Logging: eval every 250 steps = 80 points over 20k, log train loss every 10 steps
eval_interval = 250
eval_iters = 200
log_interval = 10
log_interval = 10

# weight decay
weight_decay = 1e-1

# DistributedGPT specific
model_type = 'distributed_gpt'
tp_size = 2
block_types = ['tp'] * 12  # All TP blocks = 24 communications
compile = False
