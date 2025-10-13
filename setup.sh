#!/bin/bash

echo "🚀 Setting up LLM Playground..."

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm is not installed. Please install it first:"
    echo "   npm install -g pnpm"
    exit 1
fi

# Check Node version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18 or higher is required"
    echo "   Current version: $(node -v)"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pnpm install

# Setup backend environment
if [ ! -f "packages/backend/.env" ]; then
    echo "⚙️  Creating backend .env file..."
    cp packages/backend/.env.example packages/backend/.env
    echo "⚠️  Please edit packages/backend/.env and add your API keys"
fi

# Setup frontend environment
if [ ! -f "packages/frontend/.env" ]; then
    echo "⚙️  Creating frontend .env file..."
    cp packages/frontend/.env.example packages/frontend/.env
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit packages/backend/.env and add your API keys (OpenAI or Anthropic)"
echo "   2. Run 'pnpm dev' to start the development servers"
echo "   3. Open http://localhost:5173 in your browser"
echo ""
