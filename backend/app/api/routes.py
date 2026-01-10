"""FastAPI routes for tokenization operations."""

from typing import Dict

from fastapi import APIRouter, HTTPException, status

from ..tokenization import BPETokenizer, HuggingFaceTokenizerWrapper
from ..tokenization.inspector import TokenizationInspector
from . import models

router = APIRouter(prefix="/api/v1/tokenization", tags=["tokenization"])

# Global tokenizer instances (in production, use dependency injection)
tokenizers: Dict[str, object] = {}


def get_tokenizer(tokenizer_type: str):
    """Get or create tokenizer of specified type."""
    if tokenizer_type not in tokenizers:
        if tokenizer_type == "bpe":
            tokenizers[tokenizer_type] = BPETokenizer(vocab_size=50257)
        elif tokenizer_type == "huggingface":
            tokenizers[tokenizer_type] = HuggingFaceTokenizerWrapper(
                vocab_size=50257, model_type="bpe"
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return tokenizers[tokenizer_type]


@router.post(
    "/encode",
    response_model=models.TokenizationResponse,
    summary="Tokenize text",
    description="Convert text to tokens and token IDs",
)
async def encode(request: models.TokenizationRequest) -> models.TokenizationResponse:
    """Tokenize text using specified tokenizer.

    Args:
        request: TokenizationRequest with text and tokenizer type

    Returns:
        TokenizationResponse with tokens, IDs, and analysis
    """
    try:
        tokenizer = get_tokenizer(request.tokenizer_type)
        inspector = TokenizationInspector(tokenizer)
        visualization = inspector.inspect(request.text)

        return models.TokenizationResponse(
            original_text=visualization.original_text,
            tokens=[models.Token(**t.__dict__) for t in visualization.tokens],
            token_ids=visualization.token_ids,
            summary=visualization.summary,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tokenization failed: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=models.TokenComparisonResponse,
    summary="Compare tokenizers",
    description="Compare tokenization results from two tokenizers",
)
async def compare(request: models.TokenComparisonRequest) -> models.TokenComparisonResponse:
    """Compare tokenization between two tokenizers.

    Args:
        request: TokenComparisonRequest with text and tokenizer types

    Returns:
        TokenComparisonResponse with comparison results
    """
    try:
        tokenizer1 = get_tokenizer(request.tokenizer1)
        tokenizer2 = get_tokenizer(request.tokenizer2)

        inspector = TokenizationInspector(tokenizer1)
        comparison = inspector.compare_tokenizers(request.text, tokenizer2)

        return models.TokenComparisonResponse(
            text=comparison["text"],
            tokenizer1_tokens=comparison["self"]["tokens"],
            tokenizer2_tokens=comparison["other"]["tokens"],
            tokenizer1_count=comparison["self"]["token_count"],
            tokenizer2_count=comparison["other"]["token_count"],
            difference=comparison["difference"]["token_count_diff"],
            shared_tokens=comparison["difference"]["shared_tokens"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}",
        )


@router.post(
    "/train",
    response_model=models.TokenizerTrainResponse,
    summary="Train tokenizer",
    description="Train a new tokenizer on provided texts",
)
async def train(request: models.TokenizerTrainRequest) -> models.TokenizerTrainResponse:
    """Train a new tokenizer on corpus.

    Args:
        request: TokenizerTrainRequest with training data and parameters

    Returns:
        TokenizerTrainResponse with training results
    """
    try:
        tokenizer = get_tokenizer(request.tokenizer_type)

        # Train the tokenizer
        if hasattr(tokenizer, "train"):
            tokenizer.train(
                request.texts,
                vocab_size=request.vocab_size,
                min_frequency=request.min_frequency,
                show_progress=False,
            )
        else:
            raise ValueError(
                f"Tokenizer type {request.tokenizer_type} does not support training"
            )

        return models.TokenizerTrainResponse(
            success=True,
            vocab_size=tokenizer.get_vocab_size(),
            texts_processed=len(request.texts),
            message=f"Successfully trained {request.tokenizer_type} tokenizer",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


@router.post(
    "/coverage",
    response_model=models.VocabularyCoverageResponse,
    summary="Analyze vocabulary coverage",
    description="Analyze how well tokenizer vocabulary covers a corpus",
)
async def coverage(
    request: models.VocabularyCoverageRequest,
) -> models.VocabularyCoverageResponse:
    """Analyze vocabulary coverage on corpus.

    Args:
        request: VocabularyCoverageRequest with texts to analyze

    Returns:
        VocabularyCoverageResponse with coverage statistics
    """
    try:
        tokenizer = get_tokenizer(request.tokenizer_type)
        inspector = TokenizationInspector(tokenizer)
        coverage_stats = inspector.analyze_vocabulary_coverage(request.texts)

        return models.VocabularyCoverageResponse(**coverage_stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coverage analysis failed: {str(e)}",
        )


@router.post(
    "/estimate-cost",
    response_model=models.CostEstimateResponse,
    summary="Estimate tokenization cost",
    description="Estimate the cost of tokenizing text based on token count",
)
async def estimate_cost(
    request: models.CostEstimateRequest,
) -> models.CostEstimateResponse:
    """Estimate cost of tokenizing text.

    Args:
        request: CostEstimateRequest with text and cost parameters

    Returns:
        CostEstimateResponse with cost estimates
    """
    try:
        tokenizer = get_tokenizer(request.tokenizer_type)
        inspector = TokenizationInspector(tokenizer)
        cost_estimate = inspector.estimate_cost(request.text, request.cost_per_token)

        return models.CostEstimateResponse(**cost_estimate)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost estimation failed: {str(e)}",
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if tokenization service is running",
)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "tokenization",
        "available_tokenizers": list(tokenizers.keys()) if tokenizers else [],
    }
