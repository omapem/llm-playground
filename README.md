# LLM Playground

A full-stack application for experimenting with multiple language models (OpenAI GPT, Anthropic Claude) through an interactive web interface. Built with React, TypeScript, Fastify, and Prisma.

## Features

### Core Functionality
- 🤖 **Multi-model support** - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, Claude 3.5 Sonnet, Claude 3 Opus/Haiku
- 💬 **Real-time streaming responses** - Server-Sent Events (SSE) with progressive content rendering
- 🎛️ **Adjustable parameters** - Fine-tune temperature, max tokens, and top_p for each conversation
- 💾 **Conversation persistence** - SQLite database with full conversation history and auto-save
- 📊 **Token counting & cost tracking** - Real-time token usage and cost estimation per message and conversation

### User Experience
- ⌨️ **Keyboard shortcuts** - `Ctrl+B` (sidebar), `Ctrl+K` (parameters), `Ctrl+N` (new chat), `Esc` (close panels)
- 🎨 **Modern UI** - Polished interface with Tailwind CSS, shadcn/ui components, and smooth animations
- 📝 **Markdown rendering** - Full GitHub-flavored markdown with syntax highlighting via rehype-highlight
- 📋 **Code block copy** - One-click copy functionality for code snippets with visual feedback
- 🔄 **Conversation management** - Create, rename, delete, and organize your chat history with kebab menus
- 🎯 **Empty state guidance** - Helpful suggestion cards for getting started

## Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite (build tool & dev server)
- Tailwind CSS + shadcn/ui components
- Zustand (state management)
- React Markdown + rehype-highlight (markdown rendering)
- Lucide React (icons)

**Backend:**
- Fastify + TypeScript
- Prisma ORM + SQLite
- OpenAI SDK (GPT models)
- Anthropic SDK (Claude models)
- Server-Sent Events (SSE) for streaming

**Infrastructure:**
- Turborepo (monorepo management)
- pnpm workspaces
- ESLint + Prettier (code quality)

## Prerequisites

- Node.js >= 18.0.0
- pnpm >= 8.0.0
- At least one LLM API key (OpenAI or Anthropic)

## Getting Started

### Quick Start (Recommended)

```bash
# 1. Install dependencies
pnpm install

# 2. Set up environment variables
./setup.sh
# This will copy .env.example files and prompt for API keys

# 3. Initialize the database
cd packages/backend
pnpm prisma migrate dev
pnpm prisma generate
cd ../..

# 4. Start development servers (both frontend & backend)
pnpm dev
```

The frontend will be available at http://localhost:5173 and the backend at http://localhost:3000.

### Manual Setup

If you prefer manual setup:

```bash
# 1. Install dependencies
pnpm install

# 2. Configure backend environment
cd packages/backend
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# 3. Set up the database
pnpm prisma migrate dev
pnpm prisma generate

# 4. Configure frontend environment (optional)
cd ../frontend
cp .env.example .env
# Edit .env if you need to change the API URL

# 5. Start development servers
cd ../..
pnpm dev
```

### Build for Production

```bash
pnpm build
```

## Development Commands

### Common Commands
```bash
pnpm dev              # Run both frontend and backend
pnpm build            # Build all packages
pnpm lint             # Lint all packages
pnpm format           # Format code with Prettier
pnpm typecheck        # TypeScript type checking
```

### Frontend Commands
```bash
cd packages/frontend
pnpm dev              # Vite dev server (http://localhost:5173)
pnpm build            # Production build
pnpm preview          # Preview production build
pnpm lint             # ESLint
```

### Backend Commands
```bash
cd packages/backend
pnpm dev              # Fastify with hot reload (http://localhost:3000)
pnpm build            # Compile TypeScript
pnpm start            # Run compiled version
```

### Database Commands
```bash
cd packages/backend
pnpm prisma migrate dev      # Create and apply migrations
pnpm prisma generate         # Generate Prisma client
pnpm prisma studio           # Open Prisma Studio GUI
pnpm prisma db push          # Push schema changes without migration
```

## Project Structure

```
llm-playground/
├── packages/
│   ├── frontend/              # React application
│   │   ├── src/
│   │   │   ├── components/       # UI components (shadcn/ui)
│   │   │   ├── hooks/            # Custom React hooks
│   │   │   ├── store/            # Zustand state management
│   │   │   ├── services/         # API client
│   │   │   ├── types/            # TypeScript types
│   │   │   └── lib/              # Utilities and helpers
│   │   └── package.json
│   │
│   └── backend/               # Fastify API server
│       ├── src/
│       │   ├── routes/           # API endpoints (chat, models, conversations)
│       │   ├── services/         # Business logic (LLM, database)
│       │   ├── types/            # TypeScript types
│       │   └── utils/            # Utility functions
│       ├── prisma/
│       │   ├── schema.prisma     # Database schema
│       │   ├── migrations/       # Database migrations
│       │   └── dev.db            # SQLite database (dev)
│       └── package.json
│
├── package.json               # Root package with Turborepo
├── pnpm-workspace.yaml        # pnpm workspace config
└── turbo.json                 # Turborepo build config
```

## API Endpoints

### Chat & Models
- **POST** `/api/chat` - Stream chat completions via SSE
- **GET** `/api/models` - List available models based on configured API keys

### Conversations
- **GET** `/api/conversations` - List all conversations with metadata
- **POST** `/api/conversations` - Create a new conversation
- **PATCH** `/api/conversations/:id` - Update conversation (e.g., rename)
- **DELETE** `/api/conversations/:id` - Delete conversation and all messages

### Messages
- **GET** `/api/conversations/:id/messages` - Get all messages for a conversation
- **POST** `/api/conversations/:id/messages` - Add a message to a conversation
- **PATCH** `/api/conversations/:conversationId/messages/:messageId` - Update message (content, tokens, cost)

## Environment Variables

### Backend (.env)
```bash
PORT=3000                          # Server port
NODE_ENV=development               # Environment (development/production)
OPENAI_API_KEY=sk-...             # OpenAI API key (optional)
ANTHROPIC_API_KEY=sk-ant-...      # Anthropic API key (optional)
DATABASE_URL=file:./dev.db        # SQLite database path (Prisma)
```

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:3000  # Backend API URL
```

**Note:** At least one API key (OpenAI or Anthropic) must be configured for the application to function.

## Development Status

### ✅ Completed (Phases 1-3)
- [x] Monorepo setup with TypeScript and Turborepo
- [x] Chat UI with MessageInput, MessageDisplay, ConversationList
- [x] Fastify API with SSE streaming
- [x] LLM provider abstraction (OpenAI + Anthropic)
- [x] Streaming API integration in frontend
- [x] Parameter controls UI (temperature, max_tokens, top_p)
- [x] SQLite + Prisma for conversation persistence
- [x] Token counting and cost estimation
- [x] Markdown rendering with syntax highlighting
- [x] Code block copy functionality
- [x] Conversation management (create, rename, delete)
- [x] Keyboard shortcuts
- [x] Auto-save functionality with progressive persistence

### 🚧 Future Enhancements (Phase 4+)
- [ ] System prompt editor
- [ ] Export conversations (JSON, Markdown)
- [ ] Search within conversations
- [ ] User authentication
- [ ] Multi-user support with API key management
- [ ] Rate limiting and usage quotas
- [ ] Comprehensive test coverage
- [ ] Docker containerization
- [ ] Production deployment (Railway/Render + Vercel)
- [ ] Database migration to PostgreSQL for production

## License

MIT
