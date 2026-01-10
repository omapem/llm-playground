"""FastAPI application entry point for LLM Playground backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as tokenization_router

# Create FastAPI application
app = FastAPI(
    title="LLM Playground API",
    description="Educational platform for learning Large Language Models",
    version="0.1.0",
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(tokenization_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Playground API",
        "version": "0.1.0",
        "description": "Educational platform for learning Large Language Models",
        "docs": "/docs",
        "health": "/api/v1/tokenization/health",
    }


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "llm-playground-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
