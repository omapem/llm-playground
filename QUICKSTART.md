# Quick Start Guide

Get the LLM Playground running in 5 minutes.

## Option 1: Docker Compose (Recommended)

**Easiest way to get everything running.**

### Prerequisites
- Docker and Docker Compose installed
- 5 minutes of time

### Steps

```bash
# 1. Clone/navigate to project
cd /path/to/llm-playground

# 2. Start all services
docker-compose up -d

# 3. Wait for services to start (30 seconds)
docker-compose ps

# 4. Open in browser
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

---

## Option 2: Local Development (Manual)

**More control, but requires setup.**

### Prerequisites
- Python 3.10+
- Node.js 18+
- 10 minutes of time

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev,training,inference,eval]"

# 4. Start development server
make dev
# Or: uvicorn app.main:app --reload

# Server runs at http://localhost:8000
```

### Frontend Setup (New Terminal)

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev

# Frontend runs at http://localhost:3000
```

### Access the Application

- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs
- **API Base:** http://localhost:8000/api/v1

---

## First Steps

### 1. Test the API

Visit the interactive API docs:
```
http://localhost:8000/docs
```

Try the tokenization endpoint:
- Click "POST /api/v1/tokenization/encode"
- Click "Try it out"
- Enter text: `"hello world"`
- Select tokenizer: `huggingface`
- Click "Execute"

You should see tokens and analysis.

### 2. Use the Frontend

Go to http://localhost:3000

- Enter text in the "Text to Tokenize" box
- Select tokenizer type (HuggingFace or BPE)
- Click "Tokenize"
- See tokens visualized with colors and details

### 3. Run Tests

```bash
cd backend

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_bpe_tokenizer.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### 4. Check API Health

```bash
# Get health status
curl http://localhost:8000/health

# Get tokenization service status
curl http://localhost:8000/api/v1/tokenization/health
```

---

## Common Commands

### Backend

```bash
cd backend

# Development
make dev              # Run server with hot reload
make test             # Run tests
make test-cov         # Tests with coverage
make lint             # Check code style
make format           # Auto-format code
make type-check       # Type checking
make clean            # Clean temporary files
```

### Frontend

```bash
cd frontend

npm run dev           # Development server
npm run build         # Production build
npm run type-check    # Type checking
npm run lint          # Linting
```

### Docker

```bash
docker-compose up -d              # Start all services
docker-compose down               # Stop all services
docker-compose logs -f backend    # View backend logs
docker-compose ps                 # Service status
```

---

## Troubleshooting

### Port Already in Use

If port 8000 or 3000 is already in use:

**Backend (port 8000):**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001 --reload
```

**Frontend (port 3000):**
```bash
# Find process using port 3000
lsof -i :3000

# Kill it
kill -9 <PID>

# Or configure in frontend/.env.local
PORT=3001 npm run dev
```

### Module Not Found Error

**Backend:**
```bash
# Reinstall dependencies
pip install -e ".[dev,training,inference,eval]"
```

**Frontend:**
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

### API Connection Error

Make sure backend is running:
```bash
curl http://localhost:8000/health
```

If not running, start it:
```bash
cd backend
make dev
```

### Docker Build Fails

```bash
# Clean and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Tests Failing

```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Run specific test for details
pytest tests/test_bpe_tokenizer.py::test_tokenizer_initialization -v
```

---

## API Examples

### Tokenize Text

```bash
curl -X POST "http://localhost:8000/api/v1/tokenization/encode" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "tokenizer_type": "huggingface"
  }'
```

### Compare Tokenizers

```bash
curl -X POST "http://localhost:8000/api/v1/tokenization/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "tokenizer1": "bpe",
    "tokenizer2": "huggingface"
  }'
```

### Estimate Cost

```bash
curl -X POST "http://localhost:8000/api/v1/tokenization/estimate-cost" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world test",
    "tokenizer_type": "huggingface",
    "cost_per_token": 0.0001
  }'
```

---

## Next Steps

1. **Explore the Code**
   - Backend: `backend/app/tokenization/`
   - Frontend: `frontend/src/components/`
   - API: `backend/app/api/routes.py`

2. **Read Documentation**
   - [Complete Tokenization Guide](./docs/TOKENIZATION.md)
   - [Development Guide](./CLAUDE.md)
   - [Project Requirements](./prd.md)

3. **Extend the Project**
   - Add new tokenizer type
   - Create additional analysis tools
   - Build training pipeline (Phase 2)

4. **Deploy**
   - For production, see deployment section in README.md
   - Use Docker images for cloud deployment
   - Configure environment variables

---

## Getting Help

- **API Documentation:** http://localhost:8000/docs
- **Code Documentation:** See [docs/](./docs/) folder
- **Issues:** Check [README.md](./README.md) troubleshooting section
- **More Details:** Read [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)

---

**Ready to go!** 🚀

**Still have issues?** Check the [README.md](./README.md) for comprehensive documentation.
