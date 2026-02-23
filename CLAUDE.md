# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Playground** is an educational platform for learning and experimenting with Large Language Models. It covers the complete ML lifecycle: data preparation, pre-training, fine-tuning, evaluation, and interactive inference. This is a personal learning project with potential for community sharing later.

**Key Context:**

- 12-week development timeline from kickoff
- Single-user focus (v1)
- Local-first architecture (data stays local)
- Educational focus with practical tooling
- Hands-on learning through building, not just using

## Architecture Overview

### High-Level Structure

The platform follows a monolithic architecture split into three main layers:

```
Frontend (Next.js/React)
    ↓
Backend API (FastAPI)
    ├── Inference Engine (vLLM/TGI)
    ├── Training Orchestrator (PyTorch + Accelerate)
    ├── Evaluation Framework (lm-eval-harness)
    └── Data Pipeline
    ↓
Data Layer (Models, Datasets, Logs)
```

### Core Components

1. **Foundation Layer** - Core ML components (✅ Tokenization & Architecture Complete)

   - `tokenization/` - ✅ BPE tokenizer + HuggingFace integration (COMPLETE)
   - `transformer/` - ✅ Multi-head attention, position encoding, FFN, layer norm, transformer blocks, visualizations (COMPLETE)
     - Includes: Attention mechanism, embeddings, feed-forward, layer normalization, transformer blocks, model configurator, visualization
   - `architecture/` - ✅ Model configurator (layers, heads, hidden size) with parameter analysis (COMPLETE)

2. **Pre-Training Pipeline** - Training from scratch (✅ Complete, Production-Hardened)

   - `data/` - ✅ Collection, cleaning, deduplication, filtering, custom upload (COMPLETE)
   - `training/` - ✅ Training loop, DDP foundation, checkpointing, job management (COMPLETE)
   - Real-time metrics and visualization

3. **Post-Training Pipeline** - Model optimization

   - `sft/` - ✅ Supervised fine-tuning with LoRA/QLoRA (COMPLETE)
   - `rlhf/` - Reward modeling and PPO/DPO training
   - Chat template handling

4. **Interactive Playground** - User-facing features

   - `chat/` - Multi-turn conversations with streaming
   - `decoding/` - Multiple generation strategies (greedy, beam, sampling)
   - `prompts/` - Template library and prompt engineering tools

5. **Evaluation Framework** - Model assessment

   - `eval/` - Benchmark integration (MMLU, HellaSwag, TruthfulQA, HumanEval)
   - `dashboard/` - Leaderboard, radar charts, comparisons
   - `metrics/` - Perplexity, BLEU, ROUGE, METEOR

6. **Supporting Features**
   - `models/` - Registry for local and HuggingFace models
   - `experiments/` - W&B/MLflow integration for tracking
   - `docs/` - Learning materials and architecture explanations

### Technology Stack

**Backend:**

- Python 3.10+
- FastAPI 0.104+ (API server)
- PyTorch 2.1+ with CUDA 12.1
- Accelerate + DeepSpeed (distributed training)
- Transformers + PEFT + TRL (model utilities)
- vLLM or Text Generation Inference (inference optimization)
- SQLite/PostgreSQL (data storage)
- Weights & Biases or MLflow (experiment tracking)

**Frontend:**

- React 18+
- Next.js 14+ (SSR/static generation)
- shadcn/ui (component library)
- Recharts (visualization)
- TanStack Query (state management)
- TypeScript (type safety)

**Development & Deployment:**

- Docker Compose (local development)
- Optional: Docker containers for cloud GPU instances

## Development Workflow

### Project Initialization

1. **Repository Setup**

   ```bash
   # Initialize git
   git init

   # Create backend structure
   mkdir -p backend/app/{tokenization,transformer,training,data,inference,eval,models,experiments}
   mkdir -p backend/tests
   mkdir -p backend/notebooks

   # Create frontend structure
   mkdir -p frontend/src/{components,pages,lib,types,hooks,services}
   mkdir -p frontend/public

   # Documentation
   mkdir -p docs/{tutorials,api,architecture}
   ```

2. **Configuration Files**

   - `backend/pyproject.toml` - Python dependencies and build config
   - `backend/.env.example` - Environment variables template
   - `backend/config/` - YAML configs for experiments (see PRD Appendix C)
   - `frontend/package.json` - Node dependencies
   - `frontend/next.config.js` - Next.js configuration
   - `docker-compose.yml` - Local development stack
   - `.cursorrules` or `.github/copilot-instructions.md` - AI assistant rules (if created)

3. **Development Environment**

   ```bash
   # Backend
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"  # Installs from pyproject.toml

   # Frontend
   cd frontend
   npm install
   ```

### Build and Run Commands

**Backend Development:**

```bash
cd backend
# Install dependencies
pip install -e ".[dev]"

# Run FastAPI server (development mode with auto-reload)
uvicorn app.main:app --reload

# Run specific component tests
pytest tests/tokenization/ -v
pytest tests/training/ -v
pytest tests/eval/ -v

# Run all tests with coverage
pytest --cov=app tests/

# Lint and format
ruff check app/
black app/
mypy app/  # Type checking

# Type checking (if using mypy)
mypy app/ --strict
```

**Frontend Development:**

```bash
cd frontend
# Development server (hot reload)
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint

# Run tests (if added later)
npm test
```

**Full Stack Local Development:**

```bash
# Option 1: Docker Compose (recommended)
docker-compose up -d

# Option 2: Run manually in separate terminals
# Terminal 1: Backend
cd backend && uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

### Testing Strategy

- **Unit Tests:** Test individual components (tokenizers, model layers, metrics)
- **Integration Tests:** Test pipelines (training loop with checkpointing, inference chain)
- **E2E Tests:** Test user workflows (train → evaluate → chat)

Test organization:

```
backend/tests/
├── unit/
│   ├── test_tokenization.py
│   ├── test_transformer.py
│   └── test_metrics.py
├── integration/
│   ├── test_training_pipeline.py
│   ├── test_inference_pipeline.py
│   └── test_eval_pipeline.py
└── e2e/
    └── test_user_workflows.py
```

**Test Running:**

```bash
# Run tests by category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run single test file
pytest tests/unit/test_tokenization.py::test_bpe_encoding -v

# Run with coverage report
pytest --cov=app --cov-report=html tests/
```

## Key Architecture Decisions

### 1. Training Framework: Pure PyTorch

- Start with PyTorch + Accelerate for learning value
- Defer Axolotl integration to v2
- Rationale: Educational transparency and control

### 2. Inference Optimization: vLLM

- Use vLLM for inference optimization
- Provides efficient serving, batching, and optimization
- Better community support and performance than TGI

### 3. Frontend Framework: Next.js with Client Components

- Next.js for SSR capability and file-based routing
- Emphasis on client-heavy components for rich visualizations
- TanStack Query for server state management

### 4. Experiment Tracking: Weights & Biases

- Primary choice: W&B for better visualizations
- Fallback: MLflow for self-hosted option
- Seamless integration with PyTorch training loops

### 5. Model Support Strategy (v1)

- **GPT-2:** Learning focused (lightweight, transparent)
- **Llama-2-7B:** Practical training and fine-tuning
- Add additional architectures in v2

## Development Guidelines

### Code Organization Principles

1. **Modular Components:** Each capability (tokenization, training, inference) is independently testable
2. **Configuration-Driven:** Use YAML configs (see PRD Appendix C) for experiments, not code changes
3. **Progressive Complexity:** Simple cases work before complex features (CPU → GPU, small → large models)
4. **Educational Clarity:** Code is understandable, with clear variable names and architecture

### Common Development Tasks

**Adding a New Metric:**

1. Implement in `backend/app/eval/metrics.py`
2. Write unit tests in `backend/tests/unit/test_metrics.py`
3. Integrate with evaluation dashboard in frontend
4. Document expected ranges and interpretation in docs

**Adding a New Model Architecture:**

1. Create configuration entry in `backend/app/architecture/`
2. Implement in transformer module with proper initialization
3. Test instantiation and forward pass
4. Add model card template in `backend/app/models/cards/`
5. Document in architecture tutorial

**Adding a Fine-Tuning Feature:**

1. Implement training loop in `backend/app/sft/` or `backend/app/rlhf/`
2. Ensure W&B integration for tracking
3. Support LoRA/QLoRA for efficiency
4. Create configuration template
5. Add tutorial showing end-to-end usage

**Adding a UI Component:**

1. Use shadcn/ui components as building blocks
2. Place in `frontend/src/components/` with clear naming
3. Keep components single-responsibility
4. Use TanStack Query for server state
5. Include TypeScript types for props

### Performance Targets

- **Model Load Time:** <30s for 7B models
- **First Token Latency:** <500ms (streaming inference)
- **Streaming Throughput:** <50ms per token
- **Training Throughput:** >1K tokens/sec on A100
- **UI Responsiveness:** <200ms for interactions
- **Dashboard Load:** <2s (50+ model comparisons)

## Common Workflows for Claude Code

### Starting a New Feature

1. **Understand the PRD Section:**

   - Review relevant section in prd.md
   - Check success metrics and requirements
   - Identify dependencies on other components

2. **Plan the Implementation:**

   - Sketch the data flow and dependencies
   - Identify which layer(s) need changes (backend API, frontend, both)
   - Plan test strategy upfront

3. **Implementation Sequence:**
   - Backend: Implement core logic first, add tests
   - Frontend: Build UI components and integration
   - Integration: Connect frontend to backend APIs
   - Testing: Run full test suite

### Debugging Training Issues

Common checkpoints:

- **Data Loading:** Verify tokenization, batch shape, device placement
- **Model Initialization:** Check parameter count matches config
- **Loss NaNs:** Look for gradient explosions, learning rate, mixed precision issues
- **Memory OOM:** Reduce batch size, enable gradient checkpointing
- **Slow Training:** Profile with PyTorch profiler, check data loading speed

Use W&B logs to track:

- Loss curve smoothness
- Gradient norms
- Learning rate schedule
- GPU memory usage
- Throughput trends

### Handling Model Conversions

- **HuggingFace to Local:** Use `transformers` library safetensors format
- **Quantization:** Use `bitsandbytes` for 4-bit/8-bit quantization
- **LoRA Merging:** Merge LoRA weights with base model before deployment
- **Model Cards:** Always generate documentation with model metadata

## Key Files & Directories

Critical files to understand the system:

- **prd.md** - Complete product specification and architecture
- **backend/app/main.py** - FastAPI app initialization
- **backend/app/config/** - YAML configuration files for experiments
- **frontend/src/pages/api/** - API route handlers
- **docker-compose.yml** - Local development environment

## Iteration & Deployment

### Development Cycle

1. Create feature branch: `git checkout -b feature/component-name`
2. Implement and test locally: `pytest tests/ --cov`
3. Frontend check: `npm run type-check && npm run lint`
4. Commit: Clear, descriptive messages
5. Open PR with test results and performance metrics
6. Deploy to staging (if applicable)
7. Merge to main

### Performance Validation

Before marking features complete:

- Run relevant benchmark suite
- Check memory/compute usage
- Verify UI responsiveness with Chrome DevTools
- Compare against targets in PRD

### Documentation Requirements

- API endpoints: FastAPI auto-generates OpenAPI docs
- Architecture: Explain key design decisions in docs/
- Tutorials: Step-by-step walkthroughs for major workflows
- Configuration: Document all YAML options with examples
- Code: Type hints and docstrings for public APIs (100% coverage required)

## Learning Resources Within Codebase

- **docs/tutorials/** - Step-by-step walkthroughs for major features
- **docs/architecture/** - Deep dives into key components
- **backend/notebooks/** - Jupyter notebooks showing concepts interactively
- **Code examples:** Look in tests/ for usage patterns

## Common Pitfalls to Avoid

1. **GPU Memory Management:** Always profile before scaling up
2. **Hardcoded Paths:** Use config files and environment variables
3. **Missing Type Hints:** TypeScript/Python types prevent bugs
4. **Untested Features:** Run tests before considering tasks complete
5. **Scope Creep:** Stick to PRD requirements, defer nice-to-haves
6. **Documentation Debt:** Document as you build, not after
7. **Breaking Backward Compatibility:** Plan migrations carefully for checkpoint formats

## Questions or Issues

Refer back to the PRD (prd.md) for:

- Detailed feature specifications
- Architecture diagrams
- API endpoint structures
- Configuration schema examples
- Risk mitigation strategies
- Timeline and phase breakdown

Future Claude Code instances should prioritize understanding the PRD completely before starting implementation.

## Implementation Status

### Completed Sections

#### ✅ Section 1.1: Tokenization Module

**Status:** Complete
**Date:** January 3-10, 2026
**Components:**

- BPE tokenizer implementation (educational)
- HuggingFace tokenizer wrapper (production)
- Tokenization inspector with visualizations
- API endpoints for encoding, comparison, training, coverage analysis
- Test suite with 85%+ coverage

**Files:**

- `backend/app/tokenization/` (4 modules)
- `backend/app/api/routes.py` (tokenization endpoints)
- `backend/tests/test_*.py` (tokenization tests)

#### ✅ Section 1.2: Architecture Components

**Status:** Complete
**Date:** January 10, 2026
**Components:**

- Multi-head self-attention mechanism
- Three position encoding variants (sinusoidal, learned, rotary)
- Feed-forward networks with multiple activations
- Three layer normalization variants
- Transformer blocks with residual connections
- Model configurator with parameter analysis
- Comprehensive attention visualization system
- 6 API endpoints for architecture management

**Files:**

- `backend/app/transformer/` (7 modules, ~2,100 lines)
  - `attention.py` - Multi-head attention
  - `embeddings.py` - Position encodings
  - `feed_forward.py` - FFN implementations
  - `layer_norm.py` - Normalization variants
  - `transformer_block.py` - Encoder blocks
  - `configurator.py` - Config management
  - `visualization.py` - Attention visualization
- `backend/app/api/transformer_routes.py` - API endpoints
- `backend/tests/test_transformer.py` - 40+ tests

**Success Metrics:**

- ✅ GPT-2 small instantiates (85,056,000 parameters)
- ✅ Visualizations render in 15.9ms (target: <500ms)

**Documentation:**

- `ARCHITECTURE_COMPONENTS_SUMMARY.md` - Detailed implementation
- `ARCHITECTURE_QUICK_REFERENCE.md` - Quick start guide

#### ✅ Section 2.1: Data Collection & Preparation

**Status:** Complete
**Date:** January 13, 2026
**Components:**

- DatasetLoader: Load WikiText, OpenWebText, C4 from HuggingFace Hub
- DataCleaner: PII removal (email, phone, SSN, cards), deduplication, quality filtering
- TrainValSplitter: Train/val/test splitting with reproducibility
- DataStats: Comprehensive dataset statistics and reporting
- 10+ FastAPI endpoints for REST integration

**Files:**

- `backend/app/data/` (5 modules, ~900 lines)
  - `loaders.py` - Dataset loading
  - `cleaning.py` - Text cleaning and PII removal
  - `splitter.py` - Data splitting
  - `stats.py` - Statistics computation
- `backend/app/api/data_routes.py` - API endpoints
- `backend/tests/test_data_*.py` - 150+ tests

**Success Metrics:**

- ✅ <1% duplicate rate achieved
- ✅ 1000+ docs/sec throughput (>1GB/10min projected)
- ✅ 72 core tests passing
- ✅ 90%+ code coverage

**Documentation:**

- `DATA_COLLECTION_SUMMARY.md` - Complete architecture and API documentation

#### ✅ Section 2.2: Training Engine

**Status:** Complete
**Date:** February 15, 2026
**Components:**

- Trainer: Main training orchestrator with gradient accumulation, mixed precision, W&B integration
- TrainingConfig: YAML-based configuration with validation and serialization
- CheckpointManager: Robust checkpoint save/load with automatic rotation
- Learning Rate Schedulers: Cosine, linear, and constant schedulers with warmup
- Metrics Tracking: Perplexity, gradient norms, throughput monitoring
- FastAPI Integration: REST API for training job management and monitoring

**Files:**

- `backend/app/training/` (9 modules, ~2,000+ lines)
  - `trainer.py` - Main training orchestrator
  - `config.py` - Configuration management
  - `checkpoint.py` - Checkpoint handling
  - `checkpoint_cleaner.py` - Quality-based checkpoint rotation
  - `scheduler.py` - LR schedulers
  - `metrics.py` - Metrics computation
  - `distributed.py` - DDP setup and distributed utilities
  - `train_script.py` - Standalone DDP training script
- `backend/app/api/training_routes.py` - API endpoints
- `backend/app/api/training_job_manager.py` - Job management with queueing
- `backend/app/api/persistence.py` - SQLite job persistence
- `backend/tests/integration/test_training_integration.py` - Integration tests
- `backend/tests/integration/test_api_integration.py` - API tests
- `backend/config/examples/` - Training configuration examples (4 YAML configs)

**Success Metrics:**

- ✅ Training loop with gradient accumulation and mixed precision
- ✅ Checkpoint save/resume with optimizer and scheduler state
- ✅ Multiple LR schedulers (cosine, linear, constant)
- ✅ W&B integration for experiment tracking
- ✅ 85%+ test coverage
- ✅ 15+ end-to-end integration tests

**Documentation:**

- `TRAINING_ENGINE_SUMMARY.md` - Complete implementation documentation
- `TRAINING_QUICK_REFERENCE.md` - Quick start guide
- `backend/config/examples/README.md` - Configuration guide with examples

#### ✅ Section 3.1: Supervised Fine-Tuning (SFT)

**Status:** Complete
**Date:** February 20, 2026
**Components:**

- SFTTrainer: LoRA/QLoRA fine-tuning with HuggingFace TRL
- Template System: Alpaca and Chat format templates with custom template support
- Dataset Processing: HuggingFace dataset integration with format validation
- Auto-validation callback integration
- Background job execution with cancellation support

**Files:**

- `backend/app/sft/` (4 modules)
  - `trainer.py` - SFT training orchestrator
  - `config.py` - SFT configuration management
  - `templates.py` - Chat template system
  - `dataset.py` - Dataset processing
- `backend/app/api/sft_routes.py` - SFT API endpoints
- `backend/tests/unit/test_sft_*.py` - Unit tests
- `backend/tests/integration/test_sft_*.py` - Integration tests

#### ✅ Production Readiness Hardening

**Status:** Complete
**Date:** February 22, 2026
**Components:**

19 issues resolved across Foundation, Pre-Training, and SFT pipelines:

- **Job Management:** Thread-safe cancellation (threading.Event), SQLite persistence, FIFO queueing, resource limits (CPU/GPU/concurrency)
- **Training Improvements:** Gradient checkpointing, quality-based checkpoint cleanup, training resume from checkpoint, DDP foundation for multi-GPU, distributed metrics aggregation
- **Data Pipeline:** Custom dataset upload with file validation and sanitization
- **Bug Fixes:** Static method bug in configurator.py, test assertion corrections

**Files:**

- `backend/app/training/distributed.py` - DDP setup, cleanup, reduce_mean
- `backend/app/training/checkpoint_cleaner.py` - Quality-based checkpoint rotation
- `backend/app/training/train_script.py` - Standalone DDP training script
- `backend/app/api/persistence.py` - SQLite job persistence (JobDatabase)
- `backend/app/data/loaders.py` - Custom dataset upload support
- `backend/tests/unit/test_resource_limits.py` - Resource limits tests
- `backend/tests/unit/test_checkpoint_cleanup.py` - Checkpoint cleanup tests
- `backend/tests/integration/test_job_cancellation.py` - Cancellation tests
- `backend/tests/integration/test_job_persistence.py` - Persistence tests
- `backend/tests/integration/test_job_queueing.py` - Queueing tests
- `backend/tests/integration/test_gradient_checkpointing.py` - Gradient checkpointing tests
- `backend/tests/integration/test_custom_dataset_upload.py` - Upload tests
- `backend/tests/integration/test_ddp.py` - DDP tests
- `backend/tests/integration/test_training_resume.py` - Resume tests

### Next Sections to Implement

#### Section 3.2: RLHF Training (Week 5-6)

- Reward modeling and PPO/DPO training
- Human preference data collection
- Chat template handling

#### Section 4: Evaluation Framework (Week 7-8)

- Benchmark integration
- Evaluation dashboard

#### Section 5: Interactive Playground (Week 9-10)

- Chat interface with streaming
- Generation strategies
- Prompt templates

### Key Files for New Tasks

**Architecture Overview:**

- `CLAUDE.md` (this file) - Development guide
- `prd.md` - Full product specification

**Module Reference Implementations:**

- `backend/app/transformer/configurator.py` - Shows architecture pattern
- `backend/app/api/transformer_routes.py` - Shows API pattern
- `backend/tests/test_transformer.py` - Shows testing pattern
- `backend/app/training/trainer.py` - Shows training orchestration pattern
- `backend/app/training/config.py` - Shows YAML configuration pattern
- `backend/app/training/distributed.py` - Shows DDP distributed training pattern
- `backend/app/api/persistence.py` - Shows SQLite persistence pattern
- `backend/app/sft/trainer.py` - Shows SFT fine-tuning pattern
- `backend/tests/integration/test_training_integration.py` - Shows integration testing pattern

**Configuration:**

- `backend/pyproject.toml` - Dependencies and build config
- `backend/app/main.py` - FastAPI app setup
- `backend/config/examples/` - Training configuration examples

### Development Notes

1. **Module Structure:** Each major feature gets its own directory under `backend/app/`
2. **API Pattern:** Router modules in `backend/app/api/` with routes and Pydantic models
3. **Testing:** Comprehensive test suite with unit + integration tests for each module
4. **Documentation:** Code docstrings + detailed summary documents for each section
5. **Commits:** Clean, descriptive commit messages documenting what was implemented

When implementing new sections:

- Start by creating the module directory structure
- Implement core components first
- Add API endpoints
- Write comprehensive tests
- Create documentation summary
- Make atomic commits with clear messages

### Strict TDD Workflow (Verify Every Step)

This project strictly adheres to Test-Driven Development. You must run tests at every stage. Do not assume code works; verify it via the test runner.

RED (Create & Run): Write a failing test. Run the test suite to confirm it fails and that the failure reason matches your expectations.

GREEN (Implement & Run): Write the minimum code to pass the test. Run the test suite to confirm the test passes.

REFACTOR (Clean & Run): Improve the code structure. Run the test suite again after every refactor to ensure no regressions were introduced.

Constraint: You are prohibited from moving to the next step until the current step has been verified by a terminal command execution.

### Interaction Rules for Claude

When starting a task, always ask: "Should I start by writing the test for this?"

Before providing a solution, summarize the test cases you plan to cover.

If a test fails unexpectedly, stop and analyze the failure before suggesting a fix.
