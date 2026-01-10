# LLM Playground

An educational platform for learning and experimenting with Large Language Models. Understand LLM internals while building functional AI applications.

**Status:** Foundation Phase (Tokenization & Architecture Components Complete) ✅

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
│   │   ├── tokenization/      # Tokenization module (complete)
│   │   ├── transformer/       # Transformer architecture (complete)
│   │   ├── api/               # FastAPI routes
│   │   └── main.py            # Application entry point
│   ├── tests/                 # Test suite
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

## Next Steps

### Phase 2: Pre-Training Pipeline (Week 3-4)

- Data collection and preparation
- Distributed training support
- Real-time metrics dashboard

### Phase 3: Post-Training Pipeline (Week 5-6)

- Supervised fine-tuning (SFT)
- LoRA/QLoRA support
- RLHF training

See [prd.md](./prd.md) for complete roadmap.

## Documentation

- [CLAUDE.md](./CLAUDE.md) - AI assistant development guide
- [ARCHITECTURE_COMPONENTS_SUMMARY.md](./ARCHITECTURE_COMPONENTS_SUMMARY.md) - Architecture implementation details
- [ARCHITECTURE_QUICK_REFERENCE.md](./ARCHITECTURE_QUICK_REFERENCE.md) - Quick start and examples
- [docs/TOKENIZATION.md](./docs/TOKENIZATION.md) - Tokenization module deep dive
- [prd.md](./prd.md) - Product requirements and specifications
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
**Current Phase:** Foundation - Tokenization Module (✅) & Architecture Components (✅) Complete
**Next Phase:** Pre-Training Pipeline (Week 3-4)
