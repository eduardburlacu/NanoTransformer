# train a DistributedGPT with Parallel blocks (minimal communication)
# same as train_shakespeare_char.py but with Parallel communication strategy

out_dir = 'out-shakespeare-char-parallel'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# DistributedGPT specific
model_type = 'distributed_gpt'
tp_size = 2
block_types = ['parallel'] * 6  # Parallel blocks for minimal communication
compile = False
