"""Pydantic models for API requests and responses."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TokenizationRequest(BaseModel):
    """Request to tokenize text."""

    text: str = Field(..., description="Text to tokenize", min_length=1)
    tokenizer_type: str = Field(
        "huggingface",
        description="Type of tokenizer: 'bpe' or 'huggingface'",
    )


class Token(BaseModel):
    """Representation of a token in the output."""

    id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token string")
    start: int = Field(..., description="Character start offset")
    end: int = Field(..., description="Character end offset")
    special: bool = Field(False, description="Whether this is a special token")


class TokenizationResponse(BaseModel):
    """Response from tokenization."""

    original_text: str = Field(..., description="Original input text")
    tokens: List[Token] = Field(..., description="List of tokens")
    token_ids: List[int] = Field(..., description="Token IDs")
    summary: Dict = Field(..., description="Summary statistics")


class TokenComparisonRequest(BaseModel):
    """Request to compare two tokenizers."""

    text: str = Field(..., description="Text to compare tokenization for")
    tokenizer1: str = Field("bpe", description="First tokenizer type")
    tokenizer2: str = Field("huggingface", description="Second tokenizer type")


class TokenComparisonResponse(BaseModel):
    """Response from tokenizer comparison."""

    text: str = Field(..., description="Input text")
    tokenizer1_tokens: List[str] = Field(..., description="Tokens from first tokenizer")
    tokenizer2_tokens: List[str] = Field(..., description="Tokens from second tokenizer")
    tokenizer1_count: int = Field(..., description="Token count from first tokenizer")
    tokenizer2_count: int = Field(..., description="Token count from second tokenizer")
    difference: int = Field(..., description="Token count difference")
    shared_tokens: int = Field(..., description="Number of shared tokens")


class TokenizerTrainRequest(BaseModel):
    """Request to train a tokenizer."""

    texts: List[str] = Field(..., description="Training texts", min_length=1)
    vocab_size: int = Field(50257, description="Target vocabulary size", ge=100)
    tokenizer_type: str = Field("huggingface", description="Type to train")
    min_frequency: int = Field(2, description="Minimum token frequency")


class TokenizerTrainResponse(BaseModel):
    """Response from tokenizer training."""

    success: bool = Field(..., description="Whether training succeeded")
    vocab_size: int = Field(..., description="Final vocabulary size")
    texts_processed: int = Field(..., description="Number of texts processed")
    message: str = Field(..., description="Training status message")


class VocabularyCoverageRequest(BaseModel):
    """Request to analyze vocabulary coverage."""

    texts: List[str] = Field(..., description="Texts to analyze", min_length=1)
    tokenizer_type: str = Field("huggingface", description="Tokenizer type to use")


class VocabularyCoverageResponse(BaseModel):
    """Response from vocabulary coverage analysis."""

    total_tokens: int = Field(..., description="Total tokens in corpus")
    unique_tokens: int = Field(..., description="Number of unique tokens")
    unknown_tokens: int = Field(..., description="Count of unknown tokens")
    coverage: float = Field(..., description="Coverage percentage")
    oov_rate: float = Field(..., description="Out-of-vocabulary rate")


class CostEstimateRequest(BaseModel):
    """Request to estimate tokenization cost."""

    text: str = Field(..., description="Text to estimate cost for")
    tokenizer_type: str = Field("huggingface", description="Tokenizer type to use")
    cost_per_token: float = Field(0.0001, description="Cost per token")


class CostEstimateResponse(BaseModel):
    """Response from cost estimation."""

    token_count: int = Field(..., description="Number of tokens")
    cost_per_token: float = Field(..., description="Cost per token")
    estimated_cost: float = Field(..., description="Estimated total cost")
    characters: int = Field(..., description="Character count")
    avg_characters_per_token: float = Field(
        ..., description="Average characters per token"
    )
