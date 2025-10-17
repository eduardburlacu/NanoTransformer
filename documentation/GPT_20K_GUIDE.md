# 20k Step Communication Experiment Guide

## Objective
Determine the minimum number of synchronization points needed to achieve validation loss close to 3.28 after 20k training steps on GPT-2 (125M parameters).

## Background
- **Standard GPT-2 (125M)**: Reaches val loss ~3.28 after 20k steps (~45 min on 8x H100s)
- **TP across 2 nodes**: 24 communications (2 per layer × 12 layers)

## Communication Points Breakdown

### Block Types:
- **TP Block**: 2 communications per layer (f_op + g_op in both attention and MLP)
- **SPD Block**: 1 communication per layer (single g_op at block end)
- **Parallel Block**: 0 communications per layer (only 2 total at model boundaries)

## Experiment Configurations

We test 6 different communication strategies with alternating block patterns:

| Config File | Comms | Block Configuration | Strategy |
|-------------|-------|---------------------|----------|
| `train_gpt2_24comm.py` | 24 | 12 TP | Baseline (all TP blocks) |
| `train_gpt2_18comm.py` | 18 | [TP, SPD] × 6 | 25% reduction, alternating |
| `train_gpt2_16comm.py` | 16 | [TP, SPD, SPD] × 4 | 33% reduction, alternating |
| `train_gpt2_12comm.py` | 12 | 12 SPD | **50% reduction**, all SPD |
| `train_gpt2_8comm.py` | 8 | [SPD, Parallel] × 6 | 67% reduction, alternating |
| `train_gpt2_6comm.py` | 6 | [SPD, Parallel, Parallel] × 4 | 75% reduction, alternating |

### Block Placement Strategy:
**Alternating patterns** distribute communication evenly throughout the model rather than clustering at beginning/end. This should provide:
- Better gradient flow
- More stable training
- Fairer comparison between configurations

## Running the Experiment

### Step 1: Test Configurations
```bash
python test_20k_experiment.py
```
This verifies:
- All 8 configs load correctly
- Models have ~125M parameters (within 1%)
- Forward pass works for each configuration

### Step 2: Prepare Data (one-time)
```bash
python data/openwebtext/prepare.py
```
This downloads and tokenizes OpenWebText (~54GB final size).

### Step 3: Run Training Experiments

#### Option A: Full Training (600k steps, ~4 days)
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_24comm.py
```

#### Option B: Quick Experiment (20k steps, ~45 mins each)
```bash
# Run all 6 configs for 20k steps
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_24comm.py --max_iters=20000
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_18comm.py --max_iters=20000
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_16comm.py --max_iters=20000
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_12comm.py --max_iters=20000
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_8comm.py --max_iters=20000
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_6comm.py --max_iters=20000
```

#### Option C: Priority Configs (test key hypotheses first)
```bash
# Test baseline and key milestones
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_24comm.py --max_iters=20000  # Baseline
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_12comm.py --max_iters=20000  # Half comms
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_6comm.py --max_iters=20000   # Quarter comms
```

### Step 4: Monitor on WandB
All runs log to WandB project `owt` with descriptive names:
- `gpt2-124M-24comm` (baseline)
- `gpt2-124M-18comm` (alternating TP-SPD)
- `gpt2-124M-16comm` (alternating TP-SPD-SPD)
- `gpt2-124M-12comm` (all SPD - half)
- `gpt2-124M-8comm` (alternating SPD-Parallel)
- `gpt2-124M-6comm` (alternating SPD-P-P - quarter)

## Expected Results

### Hypothesis:
1. **24 comms (TP)**: Should match baseline ~3.28
2. **18-16 comms (alternating)**: Minimal degradation, likely 3.28-3.32
3. **12 comms (SPD)**: Key test - can we maintain <3.35 with 50% fewer comms?
4. **8-6 comms (alternating)**: Moderate degradation expected, 3.35-3.45

### Key Questions to Answer:
1. **Can we achieve <3.35 with 12 communications (50% reduction)?**
2. **Can we achieve <3.40 with 6 communications (75% reduction)?**
3. **Does alternating block placement improve over clustered placement?**
4. **Is there a "cliff" where performance suddenly degrades?**

## Analysis Metrics

After training, compare:

### 1. Final Validation Loss at 20k Steps
Primary metric - how close to 3.28?

### 2. Training Curves
- Convergence speed
- Training stability
- Overfitting behavior

### 3. Communication vs Performance Tradeoff
Plot: Communication count (x-axis) vs Validation loss (y-axis)
- Is it linear, logarithmic, or cliff-like?
- Where's the "sweet spot"?

### 4. Tokens per Second (if measured)
- Does reduced communication actually speed up training?
- What's the practical speedup at 12, 6, 2 comms?

## Important Notes

- ✅ All models have ~125M parameters (architecturally similar)
- ✅ Same training hyperparameters across all configs
- ✅ Only difference is communication pattern
- ⚠️ Not mathematically equivalent to standard GPT-2 (especially Parallel blocks)
- ⚠️ This is a simulation - actual distributed performance may differ

## Next Steps After Results

Based on results, you can:

### If 12 comms works well:
- Declare victory - 50% comm reduction with minimal loss
- Test on larger models (350M, 774M)
- Benchmark actual distributed speedup

### If 6 comms shows promise:
- Fine-tune block placement
- Try different mixing strategies
- Test intermediate counts (7, 8, 9, 10, 11)

### If only 2 comms causes issues:
- Investigate why (gradient flow, representation quality)
- Try architectural modifications (residual connections, normalization)
- Test hybrid approaches

## Alternative Experiments

### Experiment 2: Block Placement
Test if placement matters:
```python
# Instead of TP at start, SPD at end, try:
block_types = ['spd']*6 + ['tp']*6  # Reverse order
block_types = ['tp', 'spd'] * 6      # Alternating
block_types = ['parallel']*4 + ['spd']*4 + ['tp']*4  # Progressive
```

### Experiment 3: Longer Training
If short runs are inconclusive, train selected configs to 100k steps for more stable comparison.

### Experiment 4: Different Model Sizes
Test if findings scale:
- 350M model (24 layers)
- 774M model (36 layers)

## Resources Needed

- **Compute**: 8x H100 GPUs (or equivalent)
- **Time**: ~4.5 hours for all 6 configs at 20k steps
- **Storage**: ~60GB for OpenWebText + ~10GB for checkpoints
- **Memory**: ~40GB per GPU for batch_size=12

## Troubleshooting

**OOM (Out of Memory)**:
```bash
# Reduce batch size
--batch_size=8 --gradient_accumulation_steps=60
```

**Slower than expected**:
```bash
# Disable compile if it causes issues
--compile=False
```

**NaN loss**:
```bash
# Reduce learning rate or add gradient clipping
--learning_rate=5e-4 --grad_clip=1.0
```

## Success Criteria

**Minimum Goal**: Achieve val loss <3.40 with 12 communications (50% reduction)

**Stretch Goal**: Achieve val loss <3.35 with 8 communications (67% reduction)

**Dream Goal**: Achieve val loss <3.30 with 6 communications (75% reduction)

## Why Alternating Patterns?

Alternating block types provides several advantages:
1. **Even distribution**: Communication points spread throughout the model
2. **Better gradient flow**: Avoids clustering comms at one end
3. **Stable training**: More balanced computational load
4. **Fair comparison**: Each config tested with optimal block placement

Good luck with your experiment! 🚀
