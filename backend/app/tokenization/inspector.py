"""Visual tokenization inspector for understanding how text is tokenized."""

from dataclasses import dataclass, asdict
from typing import List, Optional

from .base import BaseTokenizer, TokenizationResult


@dataclass
class Token:
    """Representation of a single token."""

    id: int
    text: str
    start: int
    end: int
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    special: bool = False


@dataclass
class TokenizationVisualization:
    """Complete visualization data for tokenization."""

    original_text: str
    tokens: List[Token]
    token_ids: List[int]
    summary: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_text": self.original_text,
            "tokens": [asdict(t) for t in self.tokens],
            "token_ids": self.token_ids,
            "summary": self.summary,
        }


class TokenizationInspector:
    """Inspector for detailed tokenization visualization and analysis."""

    def __init__(self, tokenizer: BaseTokenizer):
        """Initialize inspector with a tokenizer.

        Args:
            tokenizer: BaseTokenizer instance to inspect
        """
        self.tokenizer = tokenizer

    def inspect(self, text: str) -> TokenizationVisualization:
        """Create detailed visualization of tokenization.

        Args:
            text: Text to inspect

        Returns:
            TokenizationVisualization with detailed token information
        """
        result = self.tokenizer.encode(text)

        tokens = []
        for i, (token_str, token_id, (start, end)) in enumerate(
            zip(result.tokens, result.token_ids, result.offsets)
        ):
            token = Token(
                id=token_id,
                text=token_str,
                start=start,
                end=end,
                special=self._is_special_token(token_str),
            )
            tokens.append(token)

        summary = self._compute_summary(text, result)

        return TokenizationVisualization(
            original_text=text,
            tokens=tokens,
            token_ids=result.token_ids,
            summary=summary,
        )

    def compare_tokenizers(
        self, text: str, other_tokenizer: BaseTokenizer
    ) -> dict:
        """Compare tokenization between two tokenizers.

        Args:
            text: Text to tokenize
            other_tokenizer: Tokenizer to compare against

        Returns:
            Dictionary with comparison results
        """
        self_result = self.tokenizer.encode(text)
        other_result = other_tokenizer.encode(text)

        return {
            "text": text,
            "self": {
                "tokens": self_result.tokens,
                "token_count": len(self_result.tokens),
                "token_ids": self_result.token_ids,
            },
            "other": {
                "tokens": other_result.tokens,
                "token_count": len(other_result.tokens),
                "token_ids": other_result.token_ids,
            },
            "difference": {
                "token_count_diff": len(self_result.tokens) - len(other_result.tokens),
                "shared_tokens": len(
                    set(self_result.tokens) & set(other_result.tokens)
                ),
                "unique_to_self": list(
                    set(self_result.tokens) - set(other_result.tokens)
                ),
                "unique_to_other": list(
                    set(other_result.tokens) - set(self_result.tokens)
                ),
            },
        }

    def analyze_vocabulary_coverage(self, texts: List[str]) -> dict:
        """Analyze how well vocabulary covers a corpus.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with coverage statistics
        """
        total_tokens = 0
        unknown_tokens = 0
        unique_tokens = set()

        for text in texts:
            result = self.tokenizer.encode(text)
            total_tokens += len(result.tokens)
            unique_tokens.update(result.tokens)

            # Count unknown tokens
            unk_token = self.tokenizer.unk_token
            unknown_tokens += sum(1 for t in result.tokens if t == unk_token)

        return {
            "total_tokens": total_tokens,
            "unique_tokens": len(unique_tokens),
            "unknown_tokens": unknown_tokens,
            "coverage": (total_tokens - unknown_tokens) / total_tokens
            if total_tokens > 0
            else 0,
            "oov_rate": unknown_tokens / total_tokens if total_tokens > 0 else 0,
        }

    def estimate_cost(self, text: str, cost_per_token: float = 0.0001) -> dict:
        """Estimate API cost based on token count.

        Args:
            text: Text to estimate cost for
            cost_per_token: Cost per token in your currency

        Returns:
            Dictionary with cost estimates
        """
        result = self.tokenizer.encode(text)
        token_count = len(result.token_ids)

        return {
            "token_count": token_count,
            "cost_per_token": cost_per_token,
            "estimated_cost": token_count * cost_per_token,
            "characters": len(text),
            "avg_characters_per_token": len(text) / token_count
            if token_count > 0
            else 0,
        }

    def _is_special_token(self, token: str) -> bool:
        """Check if token is a special token.

        Args:
            token: Token string to check

        Returns:
            Whether token is special
        """
        special_indicators = ["[", "]", "<", ">", "</w>"]
        return any(indicator in token for indicator in special_indicators)

    def _compute_summary(self, text: str, result: TokenizationResult) -> dict:
        """Compute summary statistics about tokenization.

        Args:
            text: Original text
            result: Tokenization result

        Returns:
            Dictionary with summary statistics
        """
        token_count = len(result.tokens)
        char_count = len(text)

        special_tokens = sum(1 for t in result.tokens if self._is_special_token(t))
        regular_tokens = token_count - special_tokens

        return {
            "token_count": token_count,
            "character_count": char_count,
            "special_tokens": special_tokens,
            "regular_tokens": regular_tokens,
            "avg_token_length": char_count / token_count if token_count > 0 else 0,
            "compression_ratio": char_count / token_count if token_count > 0 else 0,
        }
