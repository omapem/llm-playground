# SFTTrainer Quick Reference

Quick start guide for using the SFT trainer.

## Installation

```bash
cd backend
pip install -e ".[training]"  # Installs peft, trl, wandb
```

## Basic Usage

### 1. Minimal Example

```python
from app.sft import SFTConfig, SFTTrainer

# Create config
config = SFTConfig(
    base_model="gpt2",
    dataset_name="tatsu-lab/alpaca",
    dataset_format="alpaca",
    output_dir="./outputs",
)

# Train
trainer = SFTTrainer(config)
result = trainer.train()
```

### 2. Load from YAML

```python
from app.sft import SFTConfig, SFTTrainer

config = SFTConfig.from_yaml('config/examples/sft_alpaca.yaml')
trainer = SFTTrainer(config)
result = trainer.train()
```

### 3. With Callbacks

```python
from app.sft import (
    SFTConfig, SFTTrainer,
    ValidationCallback, CheckpointCallback
)

config = SFTConfig.from_yaml('config.yaml')

callbacks = [
    ValidationCallback(eval_steps=100),
    CheckpointCallback(output_dir="./checkpoints", save_steps=500),
]

trainer = SFTTrainer(config, callbacks=callbacks)
result = trainer.train()
```

## Configuration Templates

### Alpaca Fine-Tuning (LoRA)

```yaml
base_model: "gpt2"
dataset_name: "tatsu-lab/alpaca"
dataset_format: "alpaca"
use_lora: true
lora_r: 8
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_epochs: 3
output_dir: "./outputs"
```

### Chat Fine-Tuning (QLoRA)

```yaml
base_model: "gpt2"
dataset_name: "HuggingFaceH4/ultrachat_200k"
dataset_format: "chat"
use_qlora: true  # 4-bit quantization
lora_r: 16
batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
output_dir: "./outputs"
save_merged_model: true
```

## Command-Line Usage

```bash
# Run training
python examples/train_sft.py --config config/examples/sft_alpaca.yaml

# Disable W&B
python examples/train_sft.py --config config.yaml --no-wandb
```

## Key Configuration Options

### Model
- `base_model`: HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
- `use_lora`: Enable LoRA (default: true)
- `use_qlora`: Enable 4-bit quantization (default: false)
- `lora_r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA alpha scaling (default: 16)

### Dataset
- `dataset_name`: HF dataset or local path
- `dataset_format`: "alpaca", "chat", or "completion"
- `validation_split`: Validation split ratio (default: 0.1)
- `max_seq_length`: Maximum sequence length (default: 2048)

### Training
- `batch_size`: Batch size per device (default: 4)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `num_epochs`: Number of epochs (default: 3)
- `mixed_precision`: "bf16", "fp16", or null (default: "bf16")

### Output
- `output_dir`: Output directory (default: "./sft_outputs")
- `save_steps`: Save checkpoint every N steps (default: 100)
- `save_merged_model`: Save merged model (default: false)

## Callbacks

### ValidationCallback
Runs validation at specified intervals:
```python
ValidationCallback(
    eval_steps=100,
    eval_on_epoch_end=True,
)
```

### CheckpointCallback
Saves checkpoints periodically:
```python
CheckpointCallback(
    output_dir="./checkpoints",
    save_steps=500,
    save_total_limit=3,
    save_best_only=False,
)
```

### WandBCallback
Logs to Weights & Biases:
```python
WandBCallback(
    project="llm-playground-sft",
    run_name="alpaca-gpt2",
    config=config.to_dict(),
    log_model=False,
)
```

## Output Files

After training, find:
```
output_dir/
├── adapter_model.safetensors    # LoRA weights
├── adapter_config.json          # LoRA config
├── tokenizer.json               # Tokenizer
├── tokenizer_config.json        # Tokenizer config
├── checkpoints/                 # Periodic checkpoints
└── merged/                      # Merged model (if enabled)
```

## Template Auto-Detection

The trainer automatically selects templates:
- `dataset_format="alpaca"` → AlpacaTemplate
- `dataset_format="chat"` → ChatTemplate (ChatML)

Override with:
```python
config.template_name = "chatml"  # or "llama2", "alpaca"
```

## Common Patterns

### Memory Optimization
Use QLoRA for large models:
```python
config.use_qlora = True
config.gradient_checkpointing = True
config.batch_size = 2
config.gradient_accumulation_steps = 8
```

### Fast Experimentation
Reduce training time for testing:
```python
config.max_steps = 100  # Override num_epochs
config.validation_split = 0.0  # Skip validation
config.save_steps = 50
```

### Production Training
Full settings for production:
```python
config.num_epochs = 3
config.validation_split = 0.05
config.save_steps = 500
config.save_merged_model = True
config.wandb_project = "production-sft"
```

## Troubleshooting

### Out of Memory
- Enable QLoRA: `use_qlora=True`
- Reduce batch size: `batch_size=1`
- Enable gradient checkpointing: `gradient_checkpointing=True`
- Reduce sequence length: `max_seq_length=512`

### Slow Training
- Disable gradient checkpointing (if memory allows)
- Increase batch size and reduce accumulation steps
- Use mixed precision: `mixed_precision="bf16"`

### Poor Results
- Increase LoRA rank: `lora_r=16` or `lora_r=32`
- Adjust learning rate: try 1e-4 to 5e-4
- Train for more epochs: `num_epochs=5`
- Check template formatting matches dataset

## Testing

```bash
# Run all SFT tests
pytest tests/unit/test_sft*.py -v

# Run trainer tests only
pytest tests/unit/test_sft_trainer.py -v

# Run integration tests (slow)
pytest tests/integration/test_sft_trainer_integration.py -v -m slow
```

## API Reference

See detailed documentation in:
- `backend/app/sft/trainer.py` - SFTTrainer class
- `backend/app/sft/config.py` - SFTConfig fields
- `SFT_TRAINER_SUMMARY.md` - Complete implementation details
