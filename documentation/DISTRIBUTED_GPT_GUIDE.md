# DistributedGPT Training Guide

This document explains how to train and benchmark the `DistributedGPT` model, which simulates tensor parallelism patterns.

## Overview

`DistributedGPT` provides three different communication strategies:
- **TP (Tensor Parallel)**: Standard Megatron-LM style, 4 communications per block (2 per sublayer)
- **SPD (Sequence Parallel Decomposition)**: 2 communications per block (1 per block)
- **Parallel**: 2 communications total (only at start/end of model)

All strategies use head parallelism and should produce identical results - they differ only in communication patterns.

## Quick Start with Shakespeare

### 1. Prepare Data
```bash
python data/shakespeare_char/prepare.py
```

### 2. Train with Standard TP (Baseline)
```bash
# Standard GPT for comparison
python train.py config/train_shakespeare_char.py

# DistributedGPT with TP blocks (should match standard GPT)
python train.py config/train_shakespeare_char_distributed.py
```

### 3. Train with SPD (1 communication per block)
```bash
python train.py config/train_shakespeare_char_spd.py
```

### 4. Train with Parallel blocks (minimal communication)
```bash
python train.py config/train_shakespeare_char_parallel.py
```

### 5. Sample from trained model
```bash
python sample.py --out_dir=out-shakespeare-char-distributed
python sample.py --out_dir=out-shakespeare-char-spd
python sample.py --out_dir=out-shakespeare-char-parallel
```

## Custom Block Configurations

You can create custom configurations by specifying `block_types` as a list:

```python
# In your config file:
model_type = 'distributed_gpt'
tp_size = 2
n_layer = 6

# Example: Alternating TP and SPD blocks
block_types = ['tp', 'spd', 'tp', 'spd', 'tp', 'spd']

# Example: All SPD
block_types = ['spd'] * 6

# Example: Parallel middle layers
block_types = ['tp', 'parallel', 'parallel', 'parallel', 'parallel', 'tp']
```

## Training GPT-2 (124M) on OpenWebText

### 1. Prepare OpenWebText dataset
```bash
python data/openwebtext/prepare.py
```
This will download and tokenize the dataset (takes a while).

### 2. Train GPT-2 with DistributedGPT

**Single GPU:**
```bash
python train.py config/train_gpt2_distributed.py
```

**Multi-GPU (DDP) - 8 GPUs:**
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_distributed.py
```

**Multi-node (2 nodes, 8 GPUs each):**
```bash
# On master node (e.g., IP 123.456.123.456):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=123.456.123.456 --master_port=1234 \
  train.py config/train_gpt2_distributed.py

# On worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=123.456.123.456 --master_port=1234 \
  train.py config/train_gpt2_distributed.py
```

### 3. Compare communication strategies

Edit `config/train_gpt2_distributed.py` and change `block_types`:

```python
# Standard TP (baseline)
block_types = None  # or ['tp'] * 12

# All SPD (1 comm per block)
block_types = ['spd'] * 12

# All Parallel (minimal comm)
block_types = ['parallel'] * 12

# Mixed strategy
block_types = ['tp'] * 4 + ['spd'] * 4 + ['parallel'] * 4
```

## Benchmarking

Compare performance between standard GPT and DistributedGPT:

```bash
# Benchmark standard GPT
python bench.py --model_type=gpt

# Benchmark DistributedGPT with TP
python bench.py --model_type=distributed_gpt --tp_size=2

# Benchmark DistributedGPT with SPD
python bench.py --model_type=distributed_gpt --tp_size=2 --block_types="['spd']*12"

# Benchmark DistributedGPT with Parallel
python bench.py --model_type=distributed_gpt --tp_size=2 --block_types="['parallel']*12"
```

## Command Line Overrides

You can override any config parameter:

```bash
# Change TP size
python train.py config/train_shakespeare_char_distributed.py --tp_size=4

# Change block types
python train.py config/train_shakespeare_char_distributed.py --block_types="['spd']*6"

# Change output directory
python train.py config/train_shakespeare_char_distributed.py --out_dir=out-custom

# Adjust learning rate and iterations
python train.py config/train_shakespeare_char_distributed.py --learning_rate=5e-4 --max_iters=10000
```

## Important Notes

1. **Compilation**: Set `compile=False` in configs as PyTorch compilation may have issues with custom autograd functions (`f_op` and `g_op`)

2. **Equivalent Results**: All three communication strategies (TP, SPD, Parallel) should produce nearly identical loss curves since they implement the same computation with different communication patterns

3. **TP_SIZE**: Currently set to 2 (simulating 2 GPUs), but this is a simulation - everything runs on a single GPU

4. **Block Types**: Must be a list with length equal to `n_layer`, or `None` for all TP blocks

5. **Device Support**: Works on CUDA, MPS (Apple Silicon), and CPU (though CPU will be slow)

## Expected Results

For Shakespeare character-level training (~3 minutes on A100):
- Standard GPT: ~1.47 validation loss
- DistributedGPT (TP): ~1.47 validation loss (should match)
- DistributedGPT (SPD): ~1.47 validation loss (should match)
- DistributedGPT (Parallel): ~1.47 validation loss (should match)

All variants should generate similar quality text samples.

## Troubleshooting

**Out of Memory**: Reduce `batch_size` or `n_layer`/`n_embd`

**Compilation errors**: Ensure `compile=False` in your config

**Wrong block_types length**: Ensure `len(block_types) == n_layer`

**Import errors**: Make sure `tensor_parallel_model.py` is in the same directory as `train.py`
