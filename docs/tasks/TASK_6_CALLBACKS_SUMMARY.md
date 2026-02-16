# Task 6: Custom Callbacks - Implementation Summary

**Date:** February 16, 2026
**Status:** ✅ Complete
**Test Coverage:** 26 tests, 100% pass rate

## Overview

Implemented custom training callbacks for the SFT (Supervised Fine-Tuning) module. These callbacks integrate with training loops to provide validation, W&B logging, and checkpointing functionality.

## Implementation Details

### Files Created

1. **`backend/app/sft/callbacks.py`** (451 lines)
   - Base callback class with lifecycle methods
   - ValidationCallback for periodic model evaluation
   - WandBCallback for experiment tracking
   - CheckpointCallback for model persistence

2. **`backend/tests/unit/test_sft_callbacks.py`** (471 lines)
   - 26 comprehensive unit tests
   - Tests for all callback features
   - Mock-based testing for external dependencies (W&B)

### Components Implemented

#### 1. SFTCallback (Base Class)

Abstract base class defining the callback interface:

```python
class SFTCallback(ABC):
    def on_train_begin(self, trainer, **kwargs)
    def on_train_end(self, trainer, **kwargs)
    def on_epoch_begin(self, trainer, epoch: int, **kwargs)
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs)
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float], **kwargs)
```

**Features:**
- Follows TRL/Transformers callback patterns
- All methods have default no-op implementations
- Extensible design for custom callbacks

#### 2. ValidationCallback

Performs validation on a held-out dataset at specified intervals.

**Key Features:**
- Run validation every N steps (configurable)
- Optional validation at epoch end
- Tracks best validation loss
- Graceful handling when no dataset provided
- Preserves model training state

**Configuration:**
```python
ValidationCallback(
    val_dataset=dataset,      # Optional validation dataset
    eval_steps=500,           # Run every N steps
    eval_on_epoch_end=True    # Run at epoch end
)
```

**Implementation Highlights:**
- Creates dataloader if trainer doesn't have one
- Properly sets model to eval mode during validation
- Restores original training mode after validation
- Handles both dict-style and tensor-style batches

#### 3. WandBCallback

Logs metrics and artifacts to Weights & Biases.

**Key Features:**
- Initialize W&B run with config
- Log step and epoch metrics
- Graceful fallback when W&B unavailable
- Optional model artifact logging
- Proper run cleanup on training end

**Configuration:**
```python
WandBCallback(
    project="llm-playground-sft",  # W&B project name
    run_name=None,                 # Auto-generated if None
    config=config_dict,            # Training config to log
    log_model=False                # Whether to log model artifacts
)
```

**Implementation Highlights:**
- Imports W&B only when needed (optional dependency)
- Comprehensive error handling for import/init failures
- Logs both training and validation metrics
- Properly finishes run to upload all data

#### 4. CheckpointCallback

Saves model checkpoints based on validation performance.

**Key Features:**
- Save every N steps (configurable)
- Optional "best model only" mode
- Automatic checkpoint rotation (maintains max N checkpoints)
- Supports both PEFT/LoRA and regular models
- Saves model state, step, and metrics

**Configuration:**
```python
CheckpointCallback(
    output_dir="./checkpoints",  # Checkpoint directory
    save_steps=500,              # Save every N steps
    save_total_limit=3,          # Max checkpoints to keep
    save_best_only=False         # Only save if val_loss improves
)
```

**Implementation Highlights:**
- Creates output directory automatically
- Detects PEFT models and uses `save_pretrained()`
- Falls back to torch.save() for regular models
- Deletes old checkpoints to manage disk space
- Tracks best validation loss for selective saving

## Test Coverage

### Test Categories

1. **Base Class Tests (2 tests)**
   - Lifecycle method existence
   - Default no-op behavior

2. **ValidationCallback Tests (7 tests)**
   - Initialization with defaults/custom values
   - Validation at step intervals
   - Validation at epoch end (enabled/disabled)
   - Best loss tracking
   - Handling missing dataset

3. **WandBCallback Tests (7 tests)**
   - Initialization with defaults/custom values
   - W&B run initialization
   - Step and epoch metric logging
   - Run finish on training end
   - Graceful handling when W&B unavailable

4. **CheckpointCallback Tests (8 tests)**
   - Initialization with defaults/custom values
   - Directory creation
   - Saving at step intervals
   - Best-only mode
   - Checkpoint rotation
   - Handling missing val_loss
   - Checkpoint path format

5. **Integration Tests (2 tests)**
   - Multiple callbacks working together
   - Complete lifecycle with all callbacks

### Test Results

```
26 passed, 1 warning in 5.96s
```

All tests pass successfully with proper mocking of external dependencies.

## Design Patterns

### 1. Lifecycle Callbacks

Follows the standard callback pattern used in TRL and Transformers:
- Clear lifecycle events (train_begin, step_end, epoch_end, train_end)
- Flexible kwargs for extensibility
- Default no-op implementations

### 2. Graceful Degradation

All callbacks handle errors gracefully:
- ValidationCallback: Works without dataset
- WandBCallback: Works when W&B unavailable
- CheckpointCallback: Handles save failures without crashing

### 3. Separation of Concerns

Each callback has a single, well-defined responsibility:
- ValidationCallback: Validation logic only
- WandBCallback: Logging only
- CheckpointCallback: Persistence only

### 4. Extensibility

The base class design makes it easy to add custom callbacks:

```python
class CustomCallback(SFTCallback):
    def on_step_end(self, trainer, step, metrics, **kwargs):
        # Custom logic here
        pass
```

## Usage Examples

### Basic Usage

```python
from app.sft.callbacks import ValidationCallback, WandBCallback, CheckpointCallback

# Create callbacks
val_callback = ValidationCallback(
    val_dataset=val_data,
    eval_steps=100
)

wandb_callback = WandBCallback(
    project="my-sft-project",
    config={"lr": 1e-4, "batch_size": 8}
)

checkpoint_callback = CheckpointCallback(
    output_dir="./checkpoints",
    save_steps=500,
    save_total_limit=3,
    save_best_only=True
)

# Add to trainer
trainer.add_callback(val_callback)
trainer.add_callback(wandb_callback)
trainer.add_callback(checkpoint_callback)
```

### Advanced Usage

```python
# Custom callback combining multiple features
class MetricsAndCheckpointCallback(SFTCallback):
    def __init__(self, output_dir: str):
        self.val_callback = ValidationCallback(eval_steps=100)
        self.checkpoint_callback = CheckpointCallback(output_dir=output_dir)

    def on_step_end(self, trainer, step, metrics, **kwargs):
        # Run validation
        self.val_callback.on_step_end(trainer, step, metrics, **kwargs)

        # Add val_loss to metrics if available
        if hasattr(self.val_callback, 'best_val_loss'):
            metrics['val_loss'] = self.val_callback.best_val_loss

        # Save checkpoint with val_loss
        self.checkpoint_callback.on_step_end(trainer, step, metrics, **kwargs)
```

## Integration with Existing Components

### SFT Module Structure

```
app/sft/
├── __init__.py           # Exports callbacks
├── config.py             # SFTConfig (Task 1) ✅
├── templates.py          # Template system (Task 2) ✅
├── lora_config.py        # LoRA configuration (Task 3) ✅
├── dataset.py            # Dataset processing (Task 5) ✅
├── callbacks.py          # Training callbacks (Task 6) ✅ NEW
└── trainer.py            # SFT trainer (Task 7) ⏳ Next
```

### Exports

Added to `app/sft/__init__.py`:

```python
from .callbacks import (
    SFTCallback,
    ValidationCallback,
    WandBCallback,
    CheckpointCallback,
)
```

## Dependencies

- **torch**: For model state management
- **wandb** (optional): For experiment tracking
- **logging**: For info/error messages

No new dependencies added - uses existing project dependencies.

## Performance Characteristics

### ValidationCallback
- Minimal overhead during training
- Validation runs only at specified intervals
- Proper GPU memory management (eval mode)

### WandBCallback
- Asynchronous logging (W&B handles buffering)
- Minimal impact on training speed
- Graceful fallback if unavailable

### CheckpointCallback
- I/O operations only at save intervals
- Efficient checkpoint rotation
- Supports both PEFT and full model saves

## Code Quality

### Documentation
- Comprehensive docstrings for all classes and methods
- Type hints throughout (100% coverage)
- Usage examples in docstrings

### Testing
- 26 unit tests covering all functionality
- Mock-based testing for external dependencies
- Edge cases handled (missing datasets, W&B errors, etc.)

### Error Handling
- Graceful degradation for missing dependencies
- Comprehensive try-except blocks
- Informative logging for errors

## Future Enhancements

Potential improvements for v2:

1. **Early Stopping Callback**: Stop training when validation plateaus
2. **TensorBoard Callback**: Alternative to W&B for logging
3. **Learning Rate Finder Callback**: Automatic LR range detection
4. **Gradient Monitoring Callback**: Track gradient norms and detect issues
5. **Model Quantization Callback**: Auto-quantize at checkpoints

## Verification Checklist

- ✅ All 26 tests pass
- ✅ TDD workflow followed (RED → GREEN → REFACTOR)
- ✅ Type hints 100% complete
- ✅ Comprehensive docstrings
- ✅ Graceful error handling
- ✅ Integration with existing SFT module
- ✅ No new dependencies introduced
- ✅ Follows project coding standards
- ✅ Ready for use in SFT trainer (Task 7)

## Next Steps

With callbacks complete, the next task is:

**Task 7: SFT Trainer** (`app/sft/trainer.py`)
- Integrate all components (config, templates, LoRA, dataset, callbacks)
- Create TRL SFTTrainer wrapper
- Implement training orchestration
- Add full end-to-end training functionality

The callback system is now ready to be integrated into the SFT trainer for complete training workflow support.
