# LLM Playground

A full-stack application for experimenting with multiple language models (OpenAI GPT, Anthropic Claude) through an interactive web interface.

## Features

- 🤖 Multi-model support (GPT-4, GPT-3.5 Turbo, Claude 3)
- 💬 Real-time streaming responses
- 🎛️ Adjustable parameters (temperature, max tokens, top_p)
- 💾 Conversation history management
- 🎨 Clean, modern UI with Tailwind CSS

## Tech Stack

**Frontend:**
- React + TypeScript
- Vite
- Tailwind CSS
- Zustand (state management)
- React Query (data fetching)

**Backend:**
- Fastify + TypeScript
- OpenAI SDK
- Anthropic SDK
- Zod (validation)

## Prerequisites

- Node.js >= 18.0.0
- pnpm >= 8.0.0
- At least one LLM API key (OpenAI or Anthropic)

## Getting Started

### 1. Clone and Install

```bash
# Install dependencies
pnpm install
```

### 2. Configure Environment Variables

**Backend:**
```bash
cd packages/backend
cp .env.example .env
# Edit .env and add your API keys
```

**Frontend:**
```bash
cd packages/frontend
cp .env.example .env
# Edit .env if you need to change the API URL
```

### 3. Run Development Servers

```bash
# From project root - runs both frontend and backend
pnpm dev

# Or run individually:
cd packages/frontend && pnpm dev  # Frontend on http://localhost:5173
cd packages/backend && pnpm dev   # Backend on http://localhost:3000
```

### 4. Build for Production

```bash
pnpm build
```

## Development Commands

```bash
# Development
pnpm dev              # Run both frontend and backend
pnpm build            # Build all packages
pnpm lint             # Lint all packages
pnpm format           # Format code with Prettier
pnpm typecheck        # TypeScript type checking

# Individual packages
cd packages/frontend
pnpm dev              # Vite dev server
pnpm build            # Production build
pnpm preview          # Preview production build

cd packages/backend
pnpm dev              # Fastify with hot reload
pnpm build            # Compile TypeScript
pnpm start            # Run compiled version
```

## Project Structure

```
llm-playground/
├── packages/
│   ├── frontend/          # React application
│   │   ├── src/
│   │   │   ├── components/   # UI components
│   │   │   ├── store/        # Zustand state management
│   │   │   ├── types/        # TypeScript types
│   │   │   └── lib/          # Utilities
│   │   └── package.json
│   │
│   └── backend/           # Fastify API server
│       ├── src/
│       │   ├── routes/       # API endpoints
│       │   ├── services/     # Business logic
│       │   ├── types/        # TypeScript types
│       │   └── middleware/   # Custom middleware
│       └── package.json
│
├── package.json           # Root package with Turbo
├── pnpm-workspace.yaml    # pnpm workspace config
└── turbo.json             # Turbo build config
```

## API Endpoints

### POST /api/chat
Stream chat completions from LLM models.

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "model": "gpt-4",
  "parameters": {
    "temperature": 0.7,
    "maxTokens": 2000,
    "topP": 1
  }
}
```

**Response:** Server-Sent Events (SSE) stream

### GET /api/models
List available LLM models based on configured API keys.

**Response:**
```json
{
  "models": [
    {
      "id": "gpt-4",
      "name": "GPT-4",
      "provider": "openai",
      "maxTokens": 8192
    }
  ]
}
```

## Environment Variables

### Backend (.env)
- `PORT` - Server port (default: 3000)
- `NODE_ENV` - Environment (development/production)
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key

### Frontend (.env)
- `VITE_API_URL` - Backend API URL (default: http://localhost:3000)

## Next Steps (Phase 2+)

- [ ] Implement streaming API integration in frontend
- [ ] Add parameter controls UI
- [ ] Set up PostgreSQL + Prisma for persistence
- [ ] Add token counting and cost estimation
- [ ] Implement system prompt editor
- [ ] Add markdown rendering for responses
- [ ] Write tests
- [ ] Deploy to production

## License

MIT
