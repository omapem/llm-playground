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

1. **Foundation Layer** - Core ML components
   - `tokenization/` - BPE tokenizer + HuggingFace integration
   - `transformer/` - Attention mechanisms, transformer blocks, visualizations
   - `architecture/` - Model configurator (layers, heads, hidden size)

2. **Pre-Training Pipeline** - Training from scratch
   - `data/` - Collection, cleaning, deduplication, filtering
   - `training/` - Main training loop, distributed training (DDP), checkpointing
   - Real-time metrics and visualization

3. **Post-Training Pipeline** - Model optimization
   - `sft/` - Supervised fine-tuning with LoRA/QLoRA
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
