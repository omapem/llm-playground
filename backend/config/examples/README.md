# Training Configuration Examples

This directory contains example YAML configurations for different training scenarios in the LLM Playground.

## Available Configurations

### 1. `gpt2_small_basic.yaml`
**Use Case:** Basic GPT-2 small training for learning and experimentation

**Model:** GPT-2 Small (117M parameters)
- 12 layers, 768 hidden size, 12 attention heads

**Training:**
- Batch size: 8 (effective: 32 with gradient accumulation)
- Learning rate: 6e-4 with cosine decay
- Steps: 10,000
- Mixed precision: BF16

**Expected Time:** ~1-2 hours on A100
**Memory:** ~8GB GPU memory

**Usage:**
```bash
python train.py --config backend/config/examples/gpt2_small_basic.yaml
```

---

### 2. `gpt2_medium_wikitext.yaml`
**Use Case:** Mid-scale pre-training on WikiText-103

**Model:** GPT-2 Medium (345M parameters)
- 24 layers, 1024 hidden size, 16 attention heads

**Training:**
- Batch size: 4 (effective: 32 with gradient accumulation)
- Learning rate: 3e-4 with cosine decay
- Steps: 50,000
- Mixed precision: BF16

**Expected Time:** ~8-10 hours on A100
**Memory:** ~16GB GPU memory

**Usage:**
```bash
python train.py --config backend/config/examples/gpt2_medium_wikitext.yaml
```

---

### 3. `gpt2_small_finetuning.yaml`
**Use Case:** Fine-tuning GPT-2 small on domain-specific data

**Model:** GPT-2 Small (117M parameters)

**Training:**
- Batch size: 8 (effective: 16 with gradient accumulation)
- Learning rate: 1e-4 with linear decay (lower for fine-tuning)
- Steps: 5,000
- Mixed precision: BF16
- More frequent validation to prevent overfitting

**Expected Time:** ~30-60 minutes on A100
**Memory:** ~8GB GPU memory

**Usage:**
```bash
python train.py --config backend/config/examples/gpt2_small_finetuning.yaml
```

---

### 4. `gpt2_tiny_debug.yaml`
**Use Case:** Fast iteration, debugging, testing training pipeline

**Model:** GPT-2 Tiny (Custom, ~15M parameters)
- 4 layers, 256 hidden size, 4 attention heads

**Training:**
- Batch size: 16
- Learning rate: 1e-3 with constant schedule
- Steps: 1,000
- Mixed precision: BF16

**Expected Time:** <5 minutes on any GPU
**Memory:** ~2GB GPU memory

**Usage:**
```bash
python train.py --config backend/config/examples/gpt2_tiny_debug.yaml
```

---

## Configuration Parameters

### Model Config
- `vocab_size`: Vocabulary size (50257 for GPT-2)
- `hidden_size`: Model dimension
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `intermediate_size`: FFN hidden dimension (typically 4 * hidden_size)
- `max_position_embeddings`: Maximum sequence length
- `activation`: Activation function (gelu, relu, gelu_approximate)
- `dropout_rate`: Dropout probability
- `position_encoding_type`: learned, sinusoidal, or rotary
- `layer_norm_type`: layer_norm, rms_norm, or group_norm

### Training Parameters
- `batch_size`: Batch size per device
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `learning_rate`: Peak learning rate
- `warmup_steps`: Number of warmup steps
- `max_steps`: Total training steps
- `weight_decay`: Weight decay for AdamW
- `max_grad_norm`: Gradient clipping threshold

### Scheduler
- `scheduler_type`: cosine, linear, or constant

### Mixed Precision
- `mixed_precision`: null, fp16, or bf16
  - `bf16`: Recommended for A100 GPUs (better stability)
  - `fp16`: For older GPUs (V100, etc.)
  - `null`: Full precision (FP32)

### Checkpointing
- `logging_steps`: Log metrics every N steps
- `save_steps`: Save checkpoint every N steps
- `eval_steps`: Run validation every N steps
- `checkpoint_dir`: Directory to save checkpoints
- `max_checkpoints_to_keep`: Keep only N most recent checkpoints

### W&B Tracking
- `wandb_project`: W&B project name (or null to disable)
- `wandb_entity`: W&B entity/username
- `wandb_run_name`: Name for this run

---

## Customizing Configurations

### Creating Your Own Config

1. Copy one of the example configs:
   ```bash
   cp backend/config/examples/gpt2_small_basic.yaml my_config.yaml
   ```

2. Edit parameters as needed:
   ```yaml
   # Adjust model size
   model_config:
     hidden_size: 512    # Smaller model
     num_layers: 8       # Fewer layers

   # Adjust training
   batch_size: 16        # Larger batch
   max_steps: 20000      # Train longer
   ```

3. Validate your config:
   ```python
   from app.training import TrainingConfig

   config = TrainingConfig.from_yaml("my_config.yaml")
   config.validate()  # Raises error if invalid
   ```

4. Run training:
   ```bash
   python train.py --config my_config.yaml
   ```

### Parameter Guidelines

**Model Size vs GPU Memory:**
| Model Size | Parameters | GPU Memory (BF16) | Recommended GPU |
|-----------|-----------|-------------------|----------------|
| Tiny | ~15M | ~2GB | Any modern GPU |
| Small | ~117M | ~8GB | RTX 3090, A100 |
| Medium | ~345M | ~16GB | A100 |
| Large | ~774M | ~32GB | A100 (80GB) |

**Batch Size Recommendations:**
- **Tiny models:** 16-32
- **Small models:** 8-16
- **Medium models:** 4-8
- **Large models:** 2-4

**Learning Rate Scaling:**
- Base LR for GPT-2 small: 6e-4
- Scale down for larger models: 3e-4 (medium), 2.5e-4 (large)
- Fine-tuning: 1e-4 to 1e-5

**Warmup Steps:**
- Short runs (<10K steps): 500-1000
- Medium runs (10K-100K steps): 2000-4000
- Long runs (>100K steps): 4000-10000

---

## Common Workflows

### Quick Test Run
```bash
# Use tiny debug config for fast iteration
python train.py --config backend/config/examples/gpt2_tiny_debug.yaml
```

### Pre-training from Scratch
```bash
# Use basic config and adjust max_steps
python train.py --config backend/config/examples/gpt2_small_basic.yaml
```

### Fine-tuning
```bash
# Use fine-tuning config
python train.py --config backend/config/examples/gpt2_small_finetuning.yaml
```

### Resume Training
```bash
# Edit config to set resume_from_checkpoint: true
python train.py --config backend/config/examples/gpt2_small_basic.yaml
```

---

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_position_embeddings`
- Enable `mixed_precision: bf16`

### Loss Not Decreasing
- Increase `learning_rate`
- Increase `warmup_steps`
- Check data quality
- Reduce `dropout_rate`

### Training Too Slow
- Enable `mixed_precision: bf16` (2x speedup)
- Increase `batch_size` (better GPU utilization)
- Reduce logging frequency

### Checkpoints Too Large
- Reduce `max_checkpoints_to_keep`
- Use smaller model
- Clear old checkpoints: `rm -rf ./checkpoints/old_run/`

---

## Additional Resources

- **Training Engine Documentation:** `/TRAINING_ENGINE_SUMMARY.md`
- **Quick Reference Guide:** `/TRAINING_QUICK_REFERENCE.md`
- **PRD Section 2.2:** Training Engine specification
- **Tests:** `backend/tests/integration/test_training_integration.py`

---

**Configuration Examples v1.0**
