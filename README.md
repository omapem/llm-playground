# LLM Playground

An educational platform for learning and experimenting with Large Language Models. Understand LLM internals while building functional AI applications.

**Status:** Foundation (1.1, 1.2) + Pre-Training (2.1, 2.2) + SFT (3.1) Complete ✅ | Production-Hardened

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)

### Local Development (Without Docker)

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,training,inference,eval]"

# Run development server
make dev
# Or: uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down
```

Services:

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Project Structure

```
llm-playground/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── tokenization/      # Tokenization module (✅ complete)
│   │   ├── transformer/       # Transformer architecture (✅ complete)
│   │   ├── data/              # Data collection & preparation (✅ complete)
│   │   ├── training/          # Training engine (✅ complete, production-hardened)
│   │   ├── sft/               # Supervised fine-tuning (✅ complete)
│   │   ├── api/               # FastAPI routes
│   │   └── main.py            # Application entry point
│   ├── config/
│   │   └── examples/          # Training configuration examples
│   ├── tests/                 # Test suite (500+ tests)
│   ├── pyproject.toml         # Dependencies
│   └── Makefile               # Development commands
│
├── frontend/                  # Next.js frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Next.js pages
│   │   ├── services/          # API client
│   │   └── types/             # TypeScript types
│   ├── package.json
│   └── tsconfig.json
│
├── docs/                      # Documentation
├── prd.md                     # Product requirements
├── CLAUDE.md                  # AI assistant guide
├── ARCHITECTURE_COMPONENTS_SUMMARY.md  # Architecture implementation details
├── ARCHITECTURE_QUICK_REFERENCE.md     # Quick start for architecture
├── DATA_COLLECTION_SUMMARY.md          # Data pipeline implementation details
├── TRAINING_ENGINE_SUMMARY.md          # Training engine implementation details
├── TRAINING_QUICK_REFERENCE.md         # Quick start for training
└── docker-compose.yml         # Local development stack
```

## Completed Features

### 1.1 Tokenization Module ✅

**Educational & Production-Ready Text Tokenization**

- **BPE Tokenizer** (Educational)

  - Byte Pair Encoding implementation from scratch
  - Transparent algorithm for learning
  - ~100 tokens/sec performance

- **HuggingFace Wrapper** (Production)

  - Fast Rust-based tokenization
  - Support for BPE, WordPiece, SentencePiece
  - ~10K+ tokens/sec on CPU

- **Tokenization Inspector**

  - Visual token analysis
  - Vocabulary coverage metrics
  - Cost estimation tools
  - Tokenizer comparison

- **API Endpoints**

  - `/tokenization/encode` - Tokenize text
  - `/tokenization/compare` - Compare tokenizers
  - `/tokenization/train` - Train custom tokenizer
  - `/tokenization/coverage` - Analyze vocabulary
  - `/tokenization/estimate-cost` - Cost estimation

- **Frontend UI**

  - Interactive tokenization inspector
  - Real-time visualization
  - Token statistics and details

- **Test Coverage**
  - Unit tests for tokenizers
  - Integration tests for pipelines
  - API endpoint tests
  - 85%+ code coverage

### Success Metrics Achieved

- ✅ Tokenizer matches HuggingFace output
- ✅ Processing speed >1K tokens/sec (achieved >10K)
- ✅ Custom vocabulary training working
- ✅ API endpoints fully functional
- ✅ Frontend UI responsive and feature-complete
- ✅ Test coverage >85%

### 1.2 Architecture Components ✅

**Educational & Production-Ready Transformer Components**

- **Attention Mechanism**

  - Multi-head self-attention with configurable heads
  - Query, Key, Value projections
  - Support for attention masks (padding and causal)
  - Efficient batched matrix operations

- **Position Encoding**

  - Sinusoidal encoding (fixed, excellent extrapolation)
  - Learned position embeddings (trainable, more expressive)
  - Rotary Position Embedding (RoPE, modern approach)

- **Feed-Forward Network**

  - Position-wise FFN with configurable expansion
  - Multiple activation functions (ReLU, GELU, GELU-approximate)
  - Gated Linear Unit (GLU) variant

- **Layer Normalization**

  - Standard LayerNorm with learnable scale and shift
  - RMS Normalization (efficient variant)
  - Group Normalization (for low-batch scenarios)

- **Transformer Blocks**

  - Complete encoder blocks with residual connections
  - Configurable norm placement (pre-norm or post-norm)
  - Stacked blocks for building multi-layer models

- **Model Configuration & Analysis**

  - TransformerConfig dataclass for architecture specification
  - Pre-defined configurations (GPT-2 Small/Medium, Llama-7B)
  - Parameter counting with component breakdown
  - Model size estimation and memory analysis

- **Visualization & Analysis**

  - Comprehensive attention weight visualization
  - Pattern detection (positional, distributed, local, CLS-like)
  - Entropy computation for attention focus measurement
  - Top-k connection extraction
  - Activation flow analysis

- **API Endpoints**

  - `/transformer/presets` - List available architectures
  - `/transformer/configure` - Create model configurations
  - `/transformer/parameter-count` - Analyze parameter distribution
  - `/transformer/visualize/attention` - Generate visualization data
  - `/transformer/compare` - Compare multiple architectures
  - `/transformer/health` - Service health check

- **Test Coverage**

  - 40+ unit and integration tests
  - Performance benchmarks
  - Configuration validation tests
  - All tests passing

### Architecture Success Metrics Achieved

- ✅ GPT-2 small instantiates successfully (85,056,000 parameters)
- ✅ Attention visualizations render in 15.9ms (target: <500ms)
- ✅ All components fully functional and tested
- ✅ Complete API integration with Pydantic models
- ✅ Comprehensive documentation with examples

### 2.1 Data Collection & Preparation ✅

**Educational & Production-Ready Data Pipeline**

- **DatasetLoader**

  - Support for 3 major datasets: WikiText, OpenWebText, C4
  - Flexible configuration with max_examples and split selection
  - Streaming support for large datasets
  - Automatic metadata extraction

- **DataCleaner**

  - PII removal (email, phone, SSN, credit cards, IP addresses)
  - MD5-based deduplication (<1% duplicates)
  - Quality filtering (length, whitespace normalization)
  - Language detection and filtering
  - Comprehensive statistics tracking

- **TrainValSplitter**

  - Flexible 2-way or 3-way splits (train/val/test)
  - Reproducible with random seed control
  - Handles edge cases and small datasets
  - HuggingFace dataset integration

- **DataStats**

  - Comprehensive dataset analysis
  - Vocabulary size, character/word/token counting
  - Language distribution tracking
  - Percentile analysis and reporting
  - Dataset comparison utilities

- **API Endpoints**

  - `/api/data/datasets/supported` - List datasets
  - `/api/data/datasets/load` - Load dataset
  - `/api/data/clean` - Clean documents
  - `/api/data/split` - Split into train/val/test
  - `/api/data/stats` - Compute statistics
  - `/api/data/pipeline` - End-to-end pipeline

- **Test Coverage**

  - 150+ test cases across 5 modules
  - 72 core tests passing
  - PII removal, deduplication, language detection
  - Performance benchmarks for success metrics
  - 90%+ code coverage

### Data Collection Success Metrics Achieved

- ✅ <1% duplicate rate (deduplication validation)
- ✅ 1000+ docs/sec throughput (1GB/10min projected)
- ✅ Memory-efficient batch processing
- ✅ Handles up to 50K documents without issues
- ✅ 150+ comprehensive test cases

### 2.2 Training Engine ✅

**Production-Ready Training Infrastructure**

- **Trainer**

  - Main training orchestrator with full pipeline
  - Gradient accumulation for larger effective batch sizes
  - Mixed precision training (FP16/BF16) for memory efficiency
  - Automatic checkpointing and resumption
  - Validation loop integration
  - Weights & Biases experiment tracking

- **TrainingConfig**

  - YAML-based configuration management
  - Comprehensive parameter validation
  - Serialization/deserialization support
  - Pre-configured examples for common scenarios

- **CheckpointManager**

  - Robust checkpoint save/load with atomic writes
  - Automatic checkpoint rotation (disk space management)
  - Complete state persistence (model, optimizer, scheduler)
  - Resume training from any checkpoint

- **Learning Rate Schedulers**

  - Cosine annealing with warmup (smooth decay)
  - Linear decay with warmup (predictable)
  - Constant with warmup (fine-tuning)
  - State persistence for checkpoint compatibility

- **Metrics Tracking**

  - Perplexity computation with overflow protection
  - Gradient norm tracking for stability monitoring
  - GPU memory usage tracking
  - Throughput computation (steps/sec, tokens/sec)
  - Windowed averaging for smooth metrics

- **FastAPI Integration**

  - Training job management (start/stop/status)
  - Configuration management (CRUD operations)
  - Real-time metrics API
  - Checkpoint operations API
  - Background job execution

- **Configuration Examples**

  - GPT-2 Small basic training
  - GPT-2 Medium pre-training
  - Fine-tuning configuration
  - Debug/testing setup

- **API Endpoints**

  - `/api/training/configs` - Configuration management
  - `/api/training/jobs/start` - Start training job
  - `/api/training/jobs/{id}/status` - Job status
  - `/api/training/jobs/{id}/metrics` - Current metrics
  - `/api/training/jobs/{id}/checkpoints` - Checkpoint operations

- **Test Coverage**

  - 15+ end-to-end integration tests
  - Complete training workflow tests
  - Checkpoint resumption tests
  - Mixed precision tests
  - 85%+ code coverage

### Training Engine Success Metrics Achieved

- ✅ Training loop with gradient accumulation working
- ✅ Checkpoint save/resume with exact state restoration
- ✅ Multiple LR schedulers (cosine, linear, constant)
- ✅ Mixed precision training (FP16/BF16)
- ✅ W&B integration for experiment tracking
- ✅ Real-time metrics and throughput monitoring
- ✅ Complete API integration with job management

### 3.1 Supervised Fine-Tuning (SFT) ✅

**LoRA/QLoRA Fine-Tuning Pipeline**

- **SFTTrainer**

  - LoRA and QLoRA fine-tuning with HuggingFace TRL
  - Configurable target modules, rank, and alpha
  - Automatic validation callback integration
  - Background job execution with cancellation support

- **Template System**

  - Alpaca and Chat format templates
  - Custom template support
  - Automatic dataset formatting

- **Dataset Processing**

  - HuggingFace dataset integration
  - Format validation and preprocessing
  - Train/eval split support

- **API Endpoints**

  - `/api/sft/jobs/start` - Start SFT job
  - `/api/sft/jobs/{id}/status` - Job status
  - `/api/sft/jobs/{id}/stop` - Stop job
  - `/api/sft/configs` - Configuration management

### Production Readiness Hardening ✅

**19 issues resolved across all implemented sections**

- **Job Management**

  - Thread-safe job cancellation with `threading.Event`
  - SQLite-backed job persistence (survives server restarts)
  - FIFO job queueing with automatic resource-aware scheduling
  - CPU/GPU/concurrency resource limits with `psutil` monitoring

- **Training Improvements**

  - Gradient checkpointing for 40-50% memory reduction
  - Quality-based checkpoint cleanup (keeps N best by validation loss)
  - Training resume from latest or specific checkpoint
  - DDP foundation for multi-GPU training (`torchrun` support)
  - Distributed metrics aggregation across ranks

- **Data Pipeline**

  - Custom dataset upload with file validation and sanitization
  - Security hardening (size limits, format validation)

- **Bug Fixes**

  - Fixed static method bug in `configurator.py`
  - Corrected test assertions for checkpoint counts and parameter sizes

## Development Commands

### Backend

```bash
cd backend

# Install dependencies
make install

# Run development server
make dev

# Run tests
make test
make test-cov           # With coverage

# Code quality
make lint               # Check code style
make format             # Auto-format code
make type-check         # Type checking

# Clean up
make clean
```

### Frontend

```bash
cd frontend

# Development
npm run dev

# Build
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
```

## Testing

### Run All Tests

```bash
cd backend
pytest tests/ -v
```

### Test Specific Component

**Tokenization Tests**

```bash
pytest tests/test_bpe_tokenizer.py -v
pytest tests/test_huggingface_wrapper.py -v
pytest tests/test_inspector.py -v
pytest tests/test_api.py -v
```

**Architecture Tests**

```bash
pytest tests/test_transformer.py -v
pytest tests/test_transformer.py::TestAttention -v      # Specific test class
pytest tests/test_transformer.py::TestTransformerBlock -v
pytest tests/test_transformer.py::TestIntegration -v
```

**Data Collection & Preparation Tests**

```bash
pytest tests/test_data_cleaning.py -v
pytest tests/test_data_splitter.py -v
pytest tests/test_data_stats.py -v
pytest tests/test_data_loader.py -v
pytest tests/test_data_performance.py -v               # Success metrics
```

**Training Engine Tests**

```bash
pytest tests/integration/test_training_integration.py -v
pytest tests/integration/test_api_integration.py -v
pytest tests/unit/test_training_config.py -v
pytest tests/unit/test_checkpoint.py -v
pytest tests/unit/test_scheduler.py -v
pytest tests/unit/test_metrics.py -v
```

### Coverage Report

```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

**Tokenization**

- `POST /api/v1/tokenization/encode` - Tokenize text
- `POST /api/v1/tokenization/compare` - Compare two tokenizers
- `POST /api/v1/tokenization/train` - Train tokenizer
- `POST /api/v1/tokenization/coverage` - Analyze vocabulary coverage
- `POST /api/v1/tokenization/estimate-cost` - Estimate tokenization cost

**Transformer Architecture**

- `GET /api/v1/transformer/presets` - List available architectures
- `POST /api/v1/transformer/configure` - Configure model architecture
- `POST /api/v1/transformer/parameter-count` - Count model parameters
- `POST /api/v1/transformer/visualize/attention` - Generate attention visualizations
- `GET /api/v1/transformer/compare` - Compare architecture configurations
- `GET /api/v1/transformer/health` - Architecture service status

**Data Collection & Preparation**

- `GET /api/data/datasets/supported` - List supported datasets
- `POST /api/data/datasets/info` - Get dataset information
- `POST /api/data/datasets/load` - Load dataset from HuggingFace
- `POST /api/data/clean` - Clean and prepare documents
- `POST /api/data/split` - Split data into train/val/test
- `POST /api/data/stats` - Compute dataset statistics
- `POST /api/data/stats/report` - Generate detailed report
- `POST /api/data/pipeline` - Run end-to-end pipeline

**Training Engine**

- `POST /api/training/configs` - Create training configuration
- `GET /api/training/configs` - List all configurations
- `GET /api/training/configs/{id}` - Get configuration by ID
- `DELETE /api/training/configs/{id}` - Delete configuration
- `POST /api/training/jobs/start` - Start training job
- `GET /api/training/jobs` - List all training jobs
- `GET /api/training/jobs/{id}/status` - Get job status
- `POST /api/training/jobs/{id}/stop` - Stop training job
- `GET /api/training/jobs/{id}/metrics` - Get current metrics
- `GET /api/training/jobs/{id}/metrics/history` - Get metrics history
- `GET /api/training/jobs/{id}/throughput` - Get throughput stats
- `GET /api/training/jobs/{id}/checkpoints` - List checkpoints
- `POST /api/training/jobs/{id}/resume` - Resume from checkpoint

**Health**

- `GET /health` - Service health check
- `GET /api/v1/tokenization/health` - Tokenization service status
- `GET /api/v1/transformer/health` - Transformer service status

## Configuration

### Backend Environment Variables

Create `backend/.env` from template:

```bash
cp backend/.env.example backend/.env
```

Key variables:

- `ENVIRONMENT` - development/production
- `LOG_LEVEL` - DEBUG/INFO/WARNING/ERROR
- `HUGGINGFACE_TOKEN` - For HF Hub access
- `CORS_ORIGINS` - Allowed origins for frontend

### Frontend Configuration

Set API URL in `frontend/next.config.js` or environment:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

## Implementation Progress

| Section | Component | Status | Tests | Coverage |
|---------|-----------|--------|-------|----------|
| 1.1 | Tokenization | ✅ Complete | 30+ | 85%+ |
| 1.2 | Architecture | ✅ Complete | 40+ | 90%+ |
| 2.1 | Data Collection | ✅ Complete | 150+ | 90%+ |
| 2.2 | Training Engine | ✅ Production-Ready | 100+ | 85%+ |
| 3.1 | Supervised Fine-Tuning | ✅ Complete | 80+ | 85%+ |
| - | Production Readiness | ✅ Complete | 205+ | - |
| 3.2 | RLHF | 📋 Planned | - | - |

## Next Steps

### Phase 3.2: RLHF Training (Week 5-6)

- Reward modeling and PPO/DPO training
- Chat template handling
- Human preference data collection

### Phase 4: Evaluation Framework (Week 7-8)

- Benchmark integration (MMLU, HellaSwag, TruthfulQA, HumanEval)
- Evaluation dashboard with visualizations
- Model comparison and leaderboards

### Phase 5: Interactive Playground (Week 9-10)

- Chat interface with streaming
- Multiple generation strategies (greedy, beam search, sampling)
- Prompt template library

See [prd.md](./prd.md) for complete roadmap.

## Documentation

- [CLAUDE.md](./CLAUDE.md) - AI assistant development guide
- [prd.md](./prd.md) - Product requirements and specifications

**Module Documentation:**

- [ARCHITECTURE_COMPONENTS_SUMMARY.md](./ARCHITECTURE_COMPONENTS_SUMMARY.md) - Architecture implementation details
- [ARCHITECTURE_QUICK_REFERENCE.md](./ARCHITECTURE_QUICK_REFERENCE.md) - Architecture quick start
- [DATA_COLLECTION_SUMMARY.md](./DATA_COLLECTION_SUMMARY.md) - Data pipeline implementation details
- [TRAINING_ENGINE_SUMMARY.md](./TRAINING_ENGINE_SUMMARY.md) - Training engine implementation details
- [TRAINING_QUICK_REFERENCE.md](./TRAINING_QUICK_REFERENCE.md) - Training quick start and examples
- [backend/config/examples/README.md](./backend/config/examples/README.md) - Training configuration guide

**API Documentation:**

- [Backend API Docs](http://localhost:8000/docs) - Interactive Swagger UI

## Contributing

This is a personal learning project. For modifications:

1. Create a feature branch
2. Make changes
3. Run tests and linting
4. Commit with clear messages

## License

MIT

## Contact

For questions or feedback, open an issue on GitHub.

---

**Started:** December 28, 2025
**Current Phase:** Foundation (1.1, 1.2) ✅ + Pre-Training (2.1, 2.2) ✅ + SFT (3.1) ✅ + Production Hardening ✅
**Next Phase:** RLHF (3.2) - Week 5-6
