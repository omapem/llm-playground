# Task 6: Custom Callbacks - Quick Reference

## Import

```python
from app.sft.callbacks import (
    SFTCallback,
    ValidationCallback,
    WandBCallback,
    CheckpointCallback,
)
```

## ValidationCallback

Run validation during training and track best model.

```python
val_callback = ValidationCallback(
    val_dataset=val_data,         # HuggingFace Dataset or PyTorch Dataset
    eval_steps=500,               # Run validation every N steps
    eval_on_epoch_end=True        # Also run at epoch end
)
```

**Key Features:**
- Tracks `best_val_loss` automatically
- Preserves model training state
- Works without dataset (no-op)
- Logs improvements to Python logger

## WandBCallback

Log metrics to Weights & Biases.

```python
wandb_callback = WandBCallback(
    project="llm-playground-sft",  # W&B project name
    run_name="my-experiment",      # Optional run name
    config={"lr": 1e-4},          # Config dict to log
    log_model=False               # Log model artifacts?
)
```

**Key Features:**
- Auto-initializes W&B run on training start
- Logs step and epoch metrics
- Gracefully handles missing W&B installation
- Finishes run properly on training end

## CheckpointCallback

Save model checkpoints during training.

```python
checkpoint_callback = CheckpointCallback(
    output_dir="./checkpoints",   # Where to save checkpoints
    save_steps=500,               # Save every N steps
    save_total_limit=3,           # Keep only last N checkpoints
    save_best_only=False          # Only save if val_loss improves?
)
```

**Key Features:**
- Automatic checkpoint rotation
- Supports PEFT/LoRA models (uses `save_pretrained()`)
- Falls back to torch.save() for regular models
- Tracks best validation loss
- Creates directory automatically

## Custom Callback

Create your own callback by extending `SFTCallback`:

```python
class MyCallback(SFTCallback):
    def on_train_begin(self, trainer, **kwargs):
        print("Training started!")

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float], **kwargs):
        if step % 100 == 0:
            print(f"Step {step}: loss={metrics.get('loss', 0):.4f}")

    def on_train_end(self, trainer, **kwargs):
        print("Training finished!")
```

## Lifecycle Methods

All callbacks support these methods:

| Method | When Called | Arguments |
|--------|-------------|-----------|
| `on_train_begin` | Start of training | `trainer` |
| `on_epoch_begin` | Start of each epoch | `trainer, epoch` |
| `on_step_end` | After each step | `trainer, step, metrics` |
| `on_epoch_end` | End of each epoch | `trainer, epoch, metrics` |
| `on_train_end` | End of training | `trainer` |

## Usage with Trainer

```python
# Create trainer (pseudocode - actual trainer in Task 7)
trainer = SFTTrainer(config=config, ...)

# Add callbacks
trainer.add_callback(ValidationCallback(val_dataset=val_data))
trainer.add_callback(WandBCallback(project="my-project"))
trainer.add_callback(CheckpointCallback(output_dir="./checkpoints"))

# Train
trainer.train()
```

## Common Patterns

### Best Model Only + W&B Logging

```python
callbacks = [
    ValidationCallback(val_dataset=val_data, eval_steps=100),
    WandBCallback(project="best-model-search", log_model=True),
    CheckpointCallback(
        output_dir="./checkpoints",
        save_steps=100,
        save_best_only=True,  # Only save when validation improves
        save_total_limit=1    # Keep only the best checkpoint
    )
]
```

### Frequent Validation + Infrequent Checkpointing

```python
callbacks = [
    ValidationCallback(val_dataset=val_data, eval_steps=50),
    CheckpointCallback(
        output_dir="./checkpoints",
        save_steps=500,  # Save less frequently than validation
        save_total_limit=5
    )
]
```

### Validation Only (No Checkpointing)

```python
callbacks = [
    ValidationCallback(
        val_dataset=val_data,
        eval_steps=100,
        eval_on_epoch_end=True
    )
]
```

## Testing

Run callback tests:

```bash
cd backend
pytest tests/unit/test_sft_callbacks.py -v
```

Expected: 26 tests pass

## Files

- **Implementation:** `/backend/app/sft/callbacks.py` (451 lines)
- **Tests:** `/backend/tests/unit/test_sft_callbacks.py` (471 lines)
- **Exports:** `/backend/app/sft/__init__.py`

## Next Steps

Task 7 will integrate these callbacks into the SFT trainer for complete end-to-end training functionality.
