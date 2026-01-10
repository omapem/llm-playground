"""Base tokenizer interface for educational clarity."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    tokens: List[str]
    token_ids: List[int]
    offsets: List[Tuple[int, int]]  # Character offsets for each token


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    def __init__(self, vocab_size: int = 50257, unk_token: str = "[UNK]"):
        """Initialize tokenizer with basic parameters.

        Args:
            vocab_size: Size of the vocabulary
            unk_token: Token to use for unknown words
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}

    @abstractmethod
    def encode(self, text: str) -> TokenizationResult:
        """Convert text to token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            TokenizationResult with tokens, token_ids, and offsets
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        pass

    def get_vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        """Get ID for a token."""
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))

    def id_to_token(self, token_id: int) -> str:
        """Get token string from ID."""
        return self.reverse_vocab.get(token_id, self.unk_token)
