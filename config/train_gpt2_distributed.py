# config for training DistributedGPT (124M) on OpenWebText
# same as train_gpt2.py but with DistributedGPT

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# DistributedGPT specific
model_type = 'distributed_gpt'
tp_size = 2  # Tensor parallel size
block_types = None  # None = all TP blocks. Try: ['spd']*12 or ['parallel']*12
out_dir = 'out-gpt2-distributed'
compile = False  # disable for custom autograd functions
