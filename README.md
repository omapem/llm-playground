# LLM Playground

An educational platform for learning and experimenting with Large Language Models. Understand LLM internals while building functional AI applications.

**Status:** Foundation Phase (Tokenization Module Complete) ✅

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

```bash
pytest tests/test_bpe_tokenizer.py -v
pytest tests/test_huggingface_wrapper.py -v
pytest tests/test_inspector.py -v
pytest tests/test_api.py -v
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

**Health**
- `GET /health` - Service health check
- `GET /api/v1/tokenization/health` - Tokenization service status

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

**Started:** December 28, 2024
**Current Phase:** Foundation - Tokenization Module Complete ✅
**Next Phase:** Pre-Training Pipeline
