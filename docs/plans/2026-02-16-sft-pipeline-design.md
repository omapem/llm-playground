# Supervised Fine-Tuning (SFT) Pipeline Design

**Date:** 2026-02-16
**Status:** Approved
**Scope:** Section 3.1 from PRD - Post-Training Pipeline (SFT Only)

---

## Context

The LLM Playground has completed its foundation layers (tokenization, transformer architecture, data preparation, and pre-training pipeline). The next critical phase is implementing the **Post-Training Pipeline** to enable model customization through supervised fine-tuning.

This design focuses on **Section 3.1: Supervised Fine-Tuning (SFT)** from the PRD, which is marked as **Priority P0** for Week 5. RLHF (Section 3.2) will be implemented as a separate phase.

**Why SFT Matters:**
- Enables customization of pre-trained models for specific tasks
- Makes 7B+ model fine-tuning accessible on consumer GPUs (24GB) via LoRA/QLoRA
- Supports instruction-following and conversational AI applications
- Foundation for future RLHF implementation

**User Requirements:**
- Hybrid approach: TRL library core + custom educational components
- Support Alpaca instruction format and chat/conversation formats
- HuggingFace model compatibility (Llama-2, Mistral, GPT-2, etc.)
- Educational platform with practical tooling

---

## Architecture Overview

The SFT module follows the same architectural pattern as the existing training module, creating a parallel structure that maintains consistency across the codebase.

### Module Structure

```
backend/app/sft/                   # New module (mirrors training/)
├── __init__.py
├── config.py                      # SFTConfig (like TrainingConfig)
├── trainer.py                     # SFTTrainer (uses TRL's SFTTrainer internally)
├── templates.py                   # Template system (Alpaca, Chat)
├── dataset.py                     # Dataset formatters and processors
├── lora_config.py                 # LoRA/QLoRA configuration manager
└── callbacks.py                   # Custom training callbacks

backend/app/api/
├── sft_routes.py                  # REST API endpoints (mirrors training_routes.py)
└── sft_job_manager.py             # Background job management

backend/config/examples/
├── sft_alpaca_llama2.yaml         # Example: Alpaca format
└── sft_chat_mistral.yaml          # Example: Chat format

backend/tests/
├── unit/
│   ├── test_sft_config.py         # Config validation tests
│   ├── test_sft_templates.py      # Template formatting tests
│   ├── test_sft_dataset.py        # Dataset processing tests
│   ├── test_sft_lora_config.py    # LoRA config tests
│   └── test_sft_callbacks.py      # Callback tests
└── integration/
    ├── test_sft_integration.py    # End-to-end SFT training tests
    └── test_sft_api.py            # API endpoint tests
```

### Key Design Principles

1. **Consistency:** Mirrors training module's structure (config → trainer → checkpoints)
2. **Separation of concerns:** Templates, datasets, and LoRA config are independent components
3. **TRL integration:** Uses TRL's SFTTrainer internally but exposes customized interface
4. **Configuration-driven:** YAML configs control all behavior (no hardcoding)
5. **Educational + Practical:** Custom components teach concepts, TRL provides production quality

---

## Core Components

### 1. SFTConfig (`sft/config.py`)

Configuration dataclass handling all SFT settings, with YAML serialization support.

**Key Sections:**
- **Model settings:** Base model ID, device mapping
- **LoRA settings:** Rank, alpha, dropout, target modules, QLoRA option
- **Dataset settings:** Name, format, template, max length, validation split
- **Training settings:** Epochs, batch size, learning rate, scheduler
- **Hardware settings:** Mixed precision, gradient checkpointing, optimizer

**Example:**
```python
@dataclass
class SFTConfig:
    # Model settings
    base_model: str                    # HF model ID

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    use_qlora: bool = False            # 4-bit quantization

    # Dataset settings
    dataset_name: str
    dataset_format: str                # "alpaca" or "chat"
    template_name: Optional[str] = None
    max_seq_length: int = 2048
    validation_split: float = 0.05

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03

    # Hardware settings
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    device_map: str = "auto"
```

### 2. Template System (`sft/templates.py`)

Handles formatting for different instruction formats with extensible template registry.

**Components:**
- `PromptTemplate` (Abstract Base Class)
- `AlpacaTemplate` - Instruction/input/output format
- `ChatTemplate` - Multi-turn conversations with roles (ChatML, Llama-2, etc.)
- `TemplateRegistry` - Register and retrieve templates by name

**Interface:**
```python
class PromptTemplate(ABC):
    @abstractmethod
    def format(self, example: Dict) -> str:
        """Format an example into a prompt string."""
        pass

class AlpacaTemplate(PromptTemplate):
    def format(self, example: Dict) -> str:
        # Formats: instruction, input (optional), output
        # Returns: "### Instruction:\n{instruction}\n\n### Response:\n{output}"

class ChatTemplate(PromptTemplate):
    def format(self, example: Dict) -> str:
        # Formats: multi-turn conversations with roles
        # Supports ChatML, Llama-2 chat, etc.
```

### 3. Dataset Processor (`sft/dataset.py`)

Handles dataset loading from HuggingFace Hub or local files and applies template formatting.

**Key Methods:**
```python
class SFTDatasetProcessor:
    def load_and_format(
        self,
        dataset_name: str,
        format: str,
        template: PromptTemplate,
        split: str = "train"
    ) -> Dataset:
        """Load dataset and apply template formatting."""
        # 1. Load raw dataset from HF Hub or local path
        # 2. Apply template to format examples
        # 3. Return formatted dataset (tokenization handled by TRL)
```

### 4. LoRA Configuration Manager (`sft/lora_config.py`)

Creates PEFT LoRA/QLoRA configurations from SFTConfig.

**Key Methods:**
```python
class LoRAConfigManager:
    @staticmethod
    def create_lora_config(config: SFTConfig) -> LoraConfig:
        """Create PEFT LoraConfig from SFTConfig."""
        return LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules or ["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )

    @staticmethod
    def create_qlora_config(config: SFTConfig) -> BitsAndBytesConfig:
        """Create 4-bit quantization config for QLoRA."""
        # Returns BitsAndBytesConfig for 4-bit quantization
```

### 5. SFTTrainer (`sft/trainer.py`)

Main orchestrator using TRL's SFTTrainer internally, exposing consistent interface with existing Trainer.

**Key Methods:**
```python
class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.config = config
        self.template = self._load_template()
        self.dataset_processor = SFTDatasetProcessor()

    def train(self):
        # 1. Load model with LoRA config
        # 2. Process datasets with template formatting
        # 3. Create TRL SFTTrainer instance
        # 4. Run training with W&B logging
        # 5. Save merged model + adapter weights

    def _load_model(self):
        # Load base model and apply LoRA/QLoRA

    def _prepare_datasets(self):
        # Load and format train/validation datasets

    def _create_trl_trainer(self):
        # Instantiate TRL SFTTrainer with processed components
```

### 6. Custom Callbacks (`sft/callbacks.py`)

Training callbacks for validation, logging, and checkpointing.

**Callbacks:**
- `ValidationCallback` - Periodic validation during training
- `WandBCallback` - Enhanced W&B logging with SFT-specific metrics
- `CheckpointCallback` - Save best model based on validation loss

---

## Data Flow

### End-to-End Training Flow

**1. Configuration Loading**
```
YAML config file → SFTConfig.from_yaml() → Validated SFTConfig object
```

**2. Model Preparation**
```
HF model ID → Load base model → Apply LoRA config → PEFT model ready
```
- Load model from HuggingFace Hub (e.g., `meta-llama/Llama-2-7b-hf`)
- Apply LoRA/QLoRA configuration via PEFT library
- Move to appropriate device with `device_map="auto"`

**3. Dataset Processing**
```
Raw dataset → Template formatting → Tokenization → Training-ready dataset
```
- Load from HF Hub or local path
- Apply appropriate template (Alpaca or Chat)
- Example transformation:
  ```
  {"instruction": "Explain ML", "output": "ML is..."}
  ↓
  "### Instruction:\nExplain ML\n\n### Response:\nML is..."
  ```
- TRL's DataCollator handles tokenization and padding

**4. Training Loop**
```
Configured trainer → TRL SFTTrainer → Training → Checkpoints + Merged model
```
- SFTTrainer wraps TRL's SFTTrainer with custom callbacks
- W&B tracks metrics (loss, learning rate, gradient norms)
- Periodic validation on held-out data
- Saves checkpoints at specified intervals
- Final step: merge LoRA weights with base model

**5. Output Artifacts**
```
Training run produces:
- LoRA adapter weights (adapter_model.safetensors)
- Merged model (optional, for inference)
- Training logs and metrics (W&B)
- Checkpoint metadata (config, training state)
```

**6. API Flow (for job management)**
```
POST /api/sft/train → Background job starts → Job ID returned
GET /api/sft/jobs/{id} → Job status + metrics
GET /api/sft/jobs/{id}/logs → Streaming training logs
```

---

## Configuration System

### YAML Configuration Schema

Example Alpaca format configuration:

```yaml
# backend/config/examples/sft_alpaca_llama2.yaml
experiment:
  name: "llama2-7b-alpaca-sft"
  description: "Fine-tune Llama-2-7B on Alpaca instruction dataset"
  tags: ["sft", "llama2", "alpaca"]

model:
  base_model: "meta-llama/Llama-2-7b-hf"
  load_in_4bit: false              # Set true for QLoRA
  device_map: "auto"

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

dataset:
  name: "tatsu-lab/alpaca"
  format: "alpaca"
  template: "alpaca_default"
  max_seq_length: 2048
  validation_split: 0.05
  preprocessing:
    remove_columns: []

training:
  output_dir: "./outputs/sft/llama2-alpaca"
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  max_grad_norm: 1.0

  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  save_total_limit: 3

  report_to: "wandb"
  run_name: "llama2-7b-alpaca-sft"

hardware:
  mixed_precision: "bf16"
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
```

### Chat Format Example

```yaml
# backend/config/examples/sft_chat_mistral.yaml
experiment:
  name: "mistral-7b-chat-sft"

model:
  base_model: "mistralai/Mistral-7B-v0.1"

dataset:
  name: "HuggingFaceH4/ultrachat_200k"
  format: "chat"
  template: "chatml"
  max_seq_length: 2048

# ... rest similar to Alpaca example
```

---

## API Integration

### REST API Endpoints (`backend/app/api/sft_routes.py`)

**Training Job Management:**
```
POST   /api/v1/sft/train
       Body: { "config_path": "config/examples/sft_alpaca_llama2.yaml" }
       Response: { "job_id": "sft-abc123", "status": "starting" }

GET    /api/v1/sft/jobs
       Response: List of all SFT jobs with status

GET    /api/v1/sft/jobs/{job_id}
       Response: Detailed job status, metrics, progress

DELETE /api/v1/sft/jobs/{job_id}
       Action: Stop running job

GET    /api/v1/sft/jobs/{job_id}/logs
       Response: Streaming training logs

GET    /api/v1/sft/jobs/{job_id}/metrics
       Response: Current training metrics
```

**Configuration Management:**
```
GET    /api/v1/sft/configs
       Response: List available SFT config templates

POST   /api/v1/sft/configs/validate
       Body: YAML config content
       Response: Validation result

GET    /api/v1/sft/templates
       Response: Available prompt templates
```

**Model Management:**
```
GET    /api/v1/sft/models
       Response: List trained SFT models

POST   /api/v1/sft/models/{model_id}/merge
       Action: Merge LoRA adapter with base model

GET    /api/v1/sft/models/{model_id}/info
       Response: Model metadata, training config, metrics
```

### Background Job Manager (`backend/app/api/sft_job_manager.py`)

Similar to `training_job_manager.py`:
- Manages background training processes
- Tracks job status (queued, running, completed, failed)
- Stores job metadata and metrics
- Handles graceful shutdown and cleanup

### Pydantic Models

```python
class SFTTrainRequest(BaseModel):
    config_path: str
    resume_from_checkpoint: Optional[str] = None

class SFTJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: int
    total_steps: int
    metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime
```

---

## Testing Strategy

### Test Structure

Target: **85%+ code coverage** across all SFT modules

```
backend/tests/
├── unit/                          (~60-70 tests)
│   ├── test_sft_config.py
│   ├── test_sft_templates.py
│   ├── test_sft_dataset.py
│   ├── test_sft_lora_config.py
│   └── test_sft_callbacks.py
│
└── integration/                   (~15-20 tests)
    ├── test_sft_integration.py
    └── test_sft_api.py
```

### Unit Tests

**1. Config Tests (`test_sft_config.py`)**
- YAML loading and validation
- Required field enforcement
- Default value handling
- Serialization/deserialization
- Invalid config error messages

**2. Template Tests (`test_sft_templates.py`)**
- Alpaca format conversion
- Chat format conversion
- Multi-turn conversation handling
- Special token handling
- Edge cases (empty input, long text)

**3. Dataset Tests (`test_sft_dataset.py`)**
- HF dataset loading
- Template application
- Train/validation splitting
- Column filtering
- Format detection

**4. LoRA Config Tests (`test_sft_lora_config.py`)**
- PEFT LoraConfig generation
- QLoRA (4-bit) config
- Target module selection
- Parameter validation

**5. Callback Tests (`test_sft_callbacks.py`)**
- Validation callback logic
- W&B logging
- Checkpoint saving logic

### Integration Tests

**1. End-to-End Training (`test_sft_integration.py`)**
- Train tiny model (GPT-2 small) on small dataset
- LoRA training completes successfully
- Checkpoints saved correctly
- Merged model loads and generates text
- Validation runs during training
- W&B logging works (with mock)

**2. API Integration (`test_sft_api.py`)**
- Start training job via POST
- Check job status
- Retrieve metrics
- Stop job
- Error handling (invalid config, missing model)

### Test Fixtures

```python
@pytest.fixture
def sample_alpaca_data():
    return [
        {"instruction": "What is ML?", "input": "", "output": "ML is..."},
        {"instruction": "Explain AI", "input": "", "output": "AI is..."}
    ]

@pytest.fixture
def mock_sft_config():
    return SFTConfig(
        base_model="gpt2",
        dataset_name="test_dataset",
        dataset_format="alpaca",
        num_epochs=1,
        batch_size=2
    )
```

---

## Success Metrics & Validation

### Implementation Success Criteria

**1. Functional Requirements (Must-Have from PRD)**
- ✅ Support instruction-following datasets (Alpaca format)
- ✅ LoRA/QLoRA integration for efficient fine-tuning
- ✅ Template system for different task formats (Alpaca, Chat)
- ✅ Conversation format handling (chat templates)
- ✅ Validation during training

**2. Performance Targets**
- Fine-tune 7B model on 24GB consumer GPU (via QLoRA)
- Training throughput: >500 tokens/sec/GPU (with gradient accumulation)
- Memory usage: <22GB GPU RAM for 7B model with QLoRA
- API response time: <500ms for job status checks

**3. Quality Metrics**
- Test coverage: >85% for all SFT modules
- >80% accuracy on held-out instruction set
- Zero critical bugs in training loop
- All API endpoints functional with proper error handling

### End-to-End Validation Process

**Step 1: Configuration Validation**
```bash
pytest tests/unit/test_sft_config.py -v
```

**Step 2: Component Testing**
```bash
pytest tests/unit/test_sft_templates.py -v
pytest tests/unit/test_sft_dataset.py -v
pytest tests/unit/test_sft_lora_config.py -v
```

**Step 3: Integration Testing**
```bash
pytest tests/integration/test_sft_integration.py -v
```

**Step 4: API Testing**
```bash
uvicorn app.main:app --reload
pytest tests/integration/test_sft_api.py -v
```

**Step 5: Real-World Validation**
```bash
# Fine-tune actual model
curl -X POST http://localhost:8000/api/v1/sft/train \
  -H "Content-Type: application/json" \
  -d '{"config_path": "config/examples/sft_alpaca_llama2.yaml"}'
```

### Validation Checklist

- [ ] All unit tests pass (>85% coverage)
- [ ] Integration tests complete successfully
- [ ] API endpoints respond correctly
- [ ] Training runs on GPU without OOM
- [ ] Checkpoints save and load correctly
- [ ] LoRA adapters merge successfully
- [ ] W&B logging captures all metrics
- [ ] Generated text from fine-tuned model is coherent

---

## Dependencies

**Existing Libraries (already in pyproject.toml):**
- `peft>=0.7.0` - LoRA/QLoRA implementation
- `trl>=0.7.0` - SFTTrainer and utilities
- `transformers>=4.35.0` - Model loading
- `datasets>=2.14.0` - Dataset handling
- `wandb>=0.15.0` - Experiment tracking

**New Dependencies:**
- None required - all necessary libraries already installed

---

## Timeline Estimate

Following PRD timeline of **Week 5** for SFT implementation:

**Phase 1 (Days 1-2): Core Components**
- SFTConfig with YAML support
- Template system (Alpaca + Chat)
- LoRA configuration manager

**Phase 2 (Days 3-4): Training & Dataset**
- SFTTrainer implementation
- Dataset processor
- Custom callbacks

**Phase 3 (Days 5-6): API Integration**
- REST endpoints
- Job manager
- Pydantic models

**Phase 4 (Day 7): Testing & Documentation**
- Unit tests (85%+ coverage)
- Integration tests
- API tests
- Documentation and examples

---

## Future Extensions (Post-v1)

**Phase 2: RLHF (Section 3.2)**
- Reward model training
- PPO or DPO implementation
- Preference dataset support
- KL divergence monitoring

**Additional Features:**
- Multi-task fine-tuning
- Few-shot learning evaluation
- Automatic prompt template generation
- Model comparison dashboard

---

## References

- **PRD Section 3.1:** Supervised Fine-Tuning (lines 241-260)
- **PRD Appendix C:** Configuration Schema (lines 877-908)
- **Existing Patterns:** `backend/app/training/` module structure
- **TRL Documentation:** https://huggingface.co/docs/trl
- **PEFT Documentation:** https://huggingface.co/docs/peft
