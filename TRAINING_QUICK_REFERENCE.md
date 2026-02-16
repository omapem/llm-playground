# Training Engine - Quick Reference Guide

**Quick start guide for using the Training Engine in the LLM Playground**

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Configuration Guide](#configuration-guide)
4. [Common Patterns](#common-patterns)
5. [Troubleshooting](#troubleshooting)
6. [API Reference](#api-reference)

---

## Quick Start

### Minimal Training Example
```python
from app.training import Trainer, TrainingConfig
from app.transformer import TransformerConfig
from torch.utils.data import Dataset
import torch

# 1. Create a simple dataset
class MyDataset(Dataset):
    def __init__(self, size=1000, seq_len=512):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(0, 50257, (self.seq_len,))

# 2. Configure training
config = TrainingConfig(
    model_config=TransformerConfig(
        vocab_size=50257,      # GPT-2 vocabulary
        hidden_size=768,       # Model dimension
        num_layers=12,         # Transformer layers
        num_heads=12,          # Attention heads
    ),
    batch_size=8,
    learning_rate=6e-4,
    max_steps=10000,
    checkpoint_dir="./checkpoints",
)

# 3. Create trainer and train
trainer = Trainer(
    config=config,
    train_dataset=MyDataset(size=10000),
)
trainer.train()
```

### YAML-Based Training
```yaml
# config.yaml
model_config:
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12

batch_size: 8
learning_rate: 0.0006
max_steps: 10000
checkpoint_dir: ./checkpoints
save_steps: 1000
logging_steps: 100
```

```python
from app.training import Trainer, TrainingConfig

config = TrainingConfig.from_yaml("config.yaml")
trainer = Trainer(config=config, train_dataset=my_dataset)
trainer.train()
```

---

## Core Components

### 1. Trainer
**Main training orchestrator**

```python
from app.training import Trainer

trainer = Trainer(
    config=config,                # TrainingConfig instance
    train_dataset=train_data,     # PyTorch Dataset
    val_dataset=val_data,         # Optional validation dataset
)

# Start training
trainer.train()

# Get training metrics
metrics = trainer.metrics_tracker.get_averages()
throughput = trainer.get_throughput()
```

### 2. TrainingConfig
**Configuration management**

```python
from app.training import TrainingConfig
from app.transformer import TransformerConfig

# Create config
config = TrainingConfig(
    model_config=TransformerConfig(...),
    batch_size=8,
    learning_rate=6e-4,
    max_steps=100000,
)

# Validate
config.validate()

# Save to YAML
config.to_yaml("experiment_1.yaml")

# Load from YAML
config = TrainingConfig.from_yaml("experiment_1.yaml")
```

### 3. CheckpointManager
**Checkpoint management**

```python
from app.training.checkpoint import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    max_checkpoints=5,  # Keep only 5 most recent
)

# Save checkpoint
path = manager.save_checkpoint(
    model, optimizer, scheduler,
    step=1000, loss=2.34, config=config
)

# Load latest checkpoint
latest = manager.get_latest_checkpoint()
step, loss, config = manager.load_checkpoint(
    latest, model, optimizer, scheduler
)

# List all checkpoints
checkpoints = manager.list_checkpoints()
```

### 4. Learning Rate Schedulers
**LR scheduling**

```python
from app.training.scheduler import get_scheduler

# Cosine scheduler
scheduler = get_scheduler(
    scheduler_type="cosine",
    optimizer=optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr_ratio=0.1,
)

# Linear scheduler
scheduler = get_scheduler(
    scheduler_type="linear",
    optimizer=optimizer,
    warmup_steps=2000,
    total_steps=100000,
)

# Constant scheduler
scheduler = get_scheduler(
    scheduler_type="constant",
    optimizer=optimizer,
    warmup_steps=2000,
)

# Update LR each step
scheduler.step()
```

### 5. Metrics Tracking
**Compute and track metrics**

```python
from app.training.metrics import (
    MetricsTracker,
    compute_perplexity,
    compute_gradient_norm,
    get_gpu_memory_usage,
)

# Create tracker
tracker = MetricsTracker(window_size=100)

# Update metrics
tracker.update({
    "loss": 2.34,
    "perplexity": compute_perplexity(2.34),
    "grad_norm": compute_gradient_norm(model),
    "lr": 0.0006,
})

# Get averages
avg = tracker.get_averages()
print(f"Avg Loss: {avg['loss']:.4f}")

# Get throughput
throughput = tracker.get_throughput(tokens_per_batch=4096)
print(f"Tokens/sec: {throughput['tokens_per_second']:.0f}")
```

---

## Configuration Guide

### Essential Parameters

```python
TrainingConfig(
    # Model architecture
    model_config=TransformerConfig(...),

    # Training basics
    batch_size=8,                      # Batch size per device
    learning_rate=6e-4,                # Initial learning rate
    max_steps=100000,                  # Total training steps
    warmup_steps=2000,                 # LR warmup steps

    # Optimizer
    optimizer_type="adamw",            # Only AdamW supported
    weight_decay=0.01,                 # Weight decay
    adam_beta1=0.9,                    # AdamW beta1
    adam_beta2=0.999,                  # AdamW beta2

    # Scheduler
    scheduler_type="cosine",           # "cosine", "linear", "constant"

    # Optimization
    gradient_accumulation_steps=4,     # Gradient accumulation
    max_grad_norm=1.0,                 # Gradient clipping
    mixed_precision="bf16",            # None, "fp16", "bf16"

    # Logging and checkpointing
    logging_steps=100,                 # Log every N steps
    save_steps=1000,                   # Save every N steps
    eval_steps=500,                    # Validate every N steps
    checkpoint_dir="./checkpoints",    # Where to save
    max_checkpoints_to_keep=5,         # Keep only N checkpoints

    # W&B tracking (optional)
    wandb_project="my-project",
    wandb_run_name="experiment-1",

    # Resume training
    resume_from_checkpoint=True,       # Load latest checkpoint
)
```

### Parameter Recommendations

| Model Size | Batch Size | Grad Accum | Learning Rate | Warmup | Mixed Precision |
|-----------|-----------|-----------|--------------|--------|----------------|
| GPT-2 Small (117M) | 8-16 | 2-4 | 6e-4 | 2000 | bf16 |
| GPT-2 Medium (345M) | 4-8 | 4-8 | 3e-4 | 2000 | bf16 |
| GPT-2 Large (774M) | 2-4 | 8-16 | 2.5e-4 | 2000 | bf16 |
| GPT-2 XL (1.5B) | 1-2 | 16-32 | 2e-4 | 2000 | bf16 |

### Scheduler Comparison

| Scheduler | Use Case | Characteristics |
|-----------|----------|----------------|
| **Cosine** | General purpose, long training | Smooth decay, prevents overshoot |
| **Linear** | Fixed-length training | Simple, predictable |
| **Constant** | Fine-tuning, short runs | No decay, stable LR |

---

## Common Patterns

### Pattern 1: Basic Training
```python
config = TrainingConfig(
    model_config=TransformerConfig(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12),
    batch_size=8,
    learning_rate=6e-4,
    max_steps=10000,
)

trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()
```

### Pattern 2: Training with Validation
```python
config = TrainingConfig(
    model_config=model_config,
    batch_size=8,
    max_steps=100000,
    eval_steps=500,  # Validate every 500 steps
)

trainer = Trainer(
    config=config,
    train_dataset=train_data,
    val_dataset=val_data,  # Add validation dataset
)
trainer.train()
```

### Pattern 3: Mixed Precision Training
```python
config = TrainingConfig(
    model_config=model_config,
    batch_size=8,
    mixed_precision="bf16",  # Enable BF16 (recommended for A100)
    # mixed_precision="fp16",  # Or FP16 (for older GPUs)
)

trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()
```

### Pattern 4: Gradient Accumulation
```python
config = TrainingConfig(
    model_config=model_config,
    batch_size=4,                    # Smaller batch per step
    gradient_accumulation_steps=8,   # Accumulate 8 steps
    # Effective batch size = 4 * 8 = 32
)

trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()
```

### Pattern 5: Resume Training
```python
# First training run
config = TrainingConfig(
    model_config=model_config,
    max_steps=10000,
    checkpoint_dir="./checkpoints/run1",
)
trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()

# Resume and continue
config = TrainingConfig.from_yaml("./checkpoints/run1/config.yaml")
config.max_steps = 20000  # Train longer
config.resume_from_checkpoint = True  # Load latest checkpoint
trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()  # Continues from step 10000
```

### Pattern 6: W&B Experiment Tracking
```python
config = TrainingConfig(
    model_config=model_config,
    batch_size=8,
    max_steps=100000,
    wandb_project="llm-playground",
    wandb_run_name="gpt2-small-wikitext-v1",
)

trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()  # Metrics automatically logged to W&B
```

### Pattern 7: Custom Scheduler
```python
from app.training.scheduler import CosineSchedulerWithWarmup

config = TrainingConfig(
    model_config=model_config,
    batch_size=8,
    learning_rate=6e-4,
    warmup_steps=2000,
    max_steps=100000,
    scheduler_type="cosine",  # Use cosine scheduler
)

trainer = Trainer(config=config, train_dataset=train_data)
trainer.train()
```

---

## Troubleshooting

### Issue: Loss becomes NaN

**Symptoms:** Loss suddenly becomes NaN during training

**Solutions:**
1. Reduce learning rate (try 1e-4 instead of 6e-4)
2. Enable gradient clipping: `max_grad_norm=1.0`
3. Check for bad data (inf/nan values in dataset)
4. Use mixed precision with care (BF16 more stable than FP16)

```python
config = TrainingConfig(
    learning_rate=1e-4,        # Lower LR
    max_grad_norm=1.0,         # Clip gradients
    mixed_precision="bf16",    # Use BF16 instead of FP16
)
```

### Issue: Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size
2. Enable gradient accumulation
3. Use mixed precision
4. Reduce model size or sequence length

```python
config = TrainingConfig(
    batch_size=4,                    # Smaller batch
    gradient_accumulation_steps=8,   # Accumulate gradients
    mixed_precision="bf16",          # Use less memory
)
```

### Issue: Slow Training

**Symptoms:** Low tokens/second throughput

**Solutions:**
1. Enable mixed precision (2x speedup)
2. Increase batch size (better GPU utilization)
3. Check data loading (set num_workers in DataLoader)
4. Profile with PyTorch profiler

```python
config = TrainingConfig(
    batch_size=16,              # Larger batch (if memory allows)
    mixed_precision="bf16",     # Enable mixed precision
)
```

### Issue: Checkpoint Not Found

**Symptoms:** "No checkpoint found to resume from" warning

**Solutions:**
1. Check checkpoint directory exists
2. Verify checkpoint files are present (`checkpoint_step_*.pt`)
3. Check file permissions

```python
import os
print(os.listdir(config.checkpoint_dir))  # List checkpoints
```

### Issue: Learning Rate Too High/Low

**Symptoms:** Loss oscillates (too high) or doesn't decrease (too low)

**Solutions:**
1. Adjust learning rate based on model size
2. Increase warmup steps for stability
3. Try different scheduler

```python
config = TrainingConfig(
    learning_rate=3e-4,      # Adjust based on model
    warmup_steps=4000,       # Longer warmup
    scheduler_type="cosine", # Smooth decay
)
```

---

## API Reference

### Trainer Methods
```python
Trainer(config, train_dataset, val_dataset=None)
├── train() -> None                    # Main training loop
├── get_throughput() -> Dict          # Get training throughput
└── metrics_tracker                   # Access MetricsTracker
```

### TrainingConfig Methods
```python
TrainingConfig(...)
├── validate() -> None                           # Validate config
├── to_dict() -> Dict                           # Convert to dict
├── from_dict(config_dict) -> TrainingConfig    # Load from dict
├── to_yaml(path) -> None                       # Save to YAML
├── from_yaml(path) -> TrainingConfig           # Load from YAML
└── get_effective_batch_size() -> int           # Calculate effective batch
```

### CheckpointManager Methods
```python
CheckpointManager(checkpoint_dir, max_checkpoints=5)
├── save_checkpoint(model, optimizer, scheduler, step, loss, config) -> str
├── load_checkpoint(path, model, optimizer, scheduler) -> Tuple[int, float, Config]
├── get_latest_checkpoint() -> Optional[str]
├── list_checkpoints() -> List[Dict]
└── _rotate_checkpoints() -> None
```

### Scheduler Methods
```python
get_scheduler(scheduler_type, optimizer, warmup_steps, total_steps=None, **kwargs)
# Returns scheduler with:
├── step() -> None                    # Update learning rate
├── state_dict() -> Dict             # Save state
└── load_state_dict(state) -> None   # Load state
```

### MetricsTracker Methods
```python
MetricsTracker(window_size=100)
├── update(metrics: Dict[str, float]) -> None
├── get_averages() -> Dict[str, float]
├── get_throughput(tokens_per_batch: int) -> Dict[str, float]
└── reset() -> None
```

### Utility Functions
```python
from app.training.metrics import (
    compute_perplexity,         # loss -> perplexity
    compute_gradient_norm,      # model -> grad norm
    get_gpu_memory_usage,       # -> memory stats
)
```

---

## REST API Endpoints

### Configuration Management
```bash
# Create config
POST /api/training/configs
{
  "name": "gpt2-small",
  "config": { ... }
}

# List configs
GET /api/training/configs

# Get config
GET /api/training/configs/{config_id}

# Delete config
DELETE /api/training/configs/{config_id}
```

### Job Management
```bash
# Start training job
POST /api/training/jobs/start
{
  "config": { ... },
  "dataset_path": "/path/to/data"
}

# List jobs
GET /api/training/jobs

# Get job status
GET /api/training/jobs/{job_id}/status

# Stop job
POST /api/training/jobs/{job_id}/stop
```

### Metrics
```bash
# Get current metrics
GET /api/training/jobs/{job_id}/metrics

# Get metrics history
GET /api/training/jobs/{job_id}/metrics/history

# Get throughput
GET /api/training/jobs/{job_id}/throughput
```

### Checkpoints
```bash
# List checkpoints
GET /api/training/jobs/{job_id}/checkpoints

# Download checkpoint
GET /api/training/checkpoints/{checkpoint_id}/download

# Resume from checkpoint
POST /api/training/jobs/{job_id}/resume
```

---

## Best Practices

### 1. Configuration Management
✅ **Do:** Use YAML files for configuration
✅ **Do:** Version control your configs
✅ **Do:** Validate configs before training
❌ **Don't:** Hardcode hyperparameters in scripts

### 2. Checkpointing
✅ **Do:** Save checkpoints frequently (every 1000 steps)
✅ **Do:** Limit max checkpoints to save disk space
✅ **Do:** Test checkpoint resumption before long runs
❌ **Don't:** Save too frequently (overhead)

### 3. Learning Rate
✅ **Do:** Use warmup for stability
✅ **Do:** Use cosine decay for long runs
✅ **Do:** Scale LR with batch size (if doubling batch, consider increasing LR)
❌ **Don't:** Use constant LR for long training

### 4. Memory Optimization
✅ **Do:** Use mixed precision (BF16)
✅ **Do:** Use gradient accumulation for larger batches
✅ **Do:** Monitor GPU memory usage
❌ **Don't:** Use FP32 unless necessary

### 5. Monitoring
✅ **Do:** Use W&B for experiment tracking
✅ **Do:** Monitor loss, perplexity, and gradient norms
✅ **Do:** Validate periodically
❌ **Don't:** Train blindly without monitoring

---

## Quick Command Reference

```bash
# Run training with YAML config
python train.py --config config.yaml

# Resume training
python train.py --config config.yaml --resume

# Check checkpoint directory
ls -lh ./checkpoints/

# Monitor training logs
tail -f training.log

# Launch W&B dashboard
wandb login
wandb sync ./wandb/

# Run tests
pytest backend/tests/integration/test_training_integration.py -v
```

---

## Example Configurations

### Small Model (Fast Iteration)
```yaml
model_config:
  vocab_size: 50257
  hidden_size: 256
  num_layers: 4
  num_heads: 4

batch_size: 16
learning_rate: 0.001
max_steps: 10000
warmup_steps: 500
mixed_precision: bf16
```

### Medium Model (Balanced)
```yaml
model_config:
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12

batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 0.0006
max_steps: 100000
warmup_steps: 2000
scheduler_type: cosine
mixed_precision: bf16
```

### Large Model (Production)
```yaml
model_config:
  vocab_size: 50257
  hidden_size: 1024
  num_layers: 24
  num_heads: 16

batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 0.00025
max_steps: 300000
warmup_steps: 4000
scheduler_type: cosine
mixed_precision: bf16
max_grad_norm: 1.0
weight_decay: 0.01
```

---

## Additional Resources

- **Full Documentation:** See `TRAINING_ENGINE_SUMMARY.md`
- **PRD Reference:** Section 2.2 - Training Engine
- **Tests:** `backend/tests/integration/test_training_integration.py`
- **Examples:** `backend/examples/training/`

---

**Quick Reference Guide v1.0 - Training Engine**
