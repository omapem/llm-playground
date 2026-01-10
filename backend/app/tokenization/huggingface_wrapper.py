"""Wrapper around HuggingFace tokenizers for practical usage.

This module provides production-ready tokenization using the fast
HuggingFace tokenizers library (implemented in Rust for performance).
"""

from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers
from tokenizers.processors import ByteLevel

from .base import BaseTokenizer, TokenizationResult


class HuggingFaceTokenizerWrapper(BaseTokenizer):
    """Production-grade tokenizer wrapper around HuggingFace tokenizers library.

    This wrapper provides:
    - Fast tokenization (implemented in Rust)
    - Support for multiple tokenization algorithms (BPE, WordPiece, SentencePiece)
    - Easy training and fine-tuning
    - Full compatibility with HuggingFace transformers
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        model_type: str = "bpe",
        unk_token: str = "[UNK]",
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize HuggingFace tokenizer wrapper.

        Args:
            vocab_size: Target vocabulary size
            model_type: Type of tokenizer ("bpe", "wordpiece", "sentencepiece")
            unk_token: Token for unknown words
            special_tokens: List of special tokens to reserve
        """
        super().__init__(vocab_size, unk_token)

        self.model_type = model_type
        self.special_tokens = special_tokens or []
        self.tokenizer: Optional[HFTokenizer] = None

        self._create_tokenizer()

    def _create_tokenizer(self) -> None:
        """Create a tokenizer with appropriate model."""
        if self.model_type == "bpe":
            self.tokenizer = HFTokenizer(models.BPE(unk_token=self.unk_token))
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self.tokenizer.post_processor = ByteLevel(trim_offsets=True)
        elif self.model_type == "wordpiece":
            self.tokenizer = HFTokenizer(models.WordPiece(unk_token=self.unk_token))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        elif self.model_type == "sentencepiece":
            # Note: SentencePiece requires separate training
            self.tokenizer = HFTokenizer(models.SentencePiece())
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
        """Train tokenizer on a corpus.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a token
            special_tokens: Special tokens to add
            show_progress: Whether to show progress
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        tokens_to_add = special_tokens or self.special_tokens
        if self.unk_token not in tokens_to_add:
            tokens_to_add = [self.unk_token] + tokens_to_add

        # Create trainer based on model type
        if self.model_type == "bpe":
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=tokens_to_add,
                show_progress=show_progress,
                continuing_subword_prefix="##",
            )
        elif self.model_type == "wordpiece":
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=tokens_to_add,
                show_progress=show_progress,
                continuing_subword_prefix="##",
            )
        else:
            raise ValueError(f"Training not supported for {self.model_type}")

        # Train on texts
        self.tokenizer.train_from_iterator(
            texts, trainer=trainer, length=len(texts)
        )

        # Update internal vocab tracking
        self._sync_vocab()

    def train_from_files(
        self,
        file_paths: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
        """Train tokenizer from files.

        Args:
            file_paths: List of file paths to train on
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a token
            special_tokens: Special tokens to add
            show_progress: Whether to show progress
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        tokens_to_add = special_tokens or self.special_tokens
        if self.unk_token not in tokens_to_add:
            tokens_to_add = [self.unk_token] + tokens_to_add

        if self.model_type == "bpe":
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=tokens_to_add,
                show_progress=show_progress,
            )
        elif self.model_type == "wordpiece":
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=tokens_to_add,
                show_progress=show_progress,
            )
        else:
            raise ValueError(f"Training not supported for {self.model_type}")

        self.tokenizer.train(file_paths, trainer=trainer)
        self._sync_vocab()

    def encode(self, text: str) -> TokenizationResult:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            TokenizationResult with tokens, token_ids, and offsets
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        encoding = self.tokenizer.encode(text)

        return TokenizationResult(
            tokens=encoding.tokens,
            token_ids=encoding.ids,
            offsets=encoding.offsets,
        )

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _sync_vocab(self) -> None:
        """Synchronize internal vocab with tokenizer."""
        if self.tokenizer is None:
            return

        hf_vocab = self.tokenizer.get_vocab()
        self.vocab = hf_vocab
        self.reverse_vocab = {v: k for k, v in hf_vocab.items()}

    def save(self, filepath: Path) -> None:
        """Save tokenizer to file.

        Args:
            filepath: Path to save tokenizer
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        self.tokenizer.save(str(filepath), pretty=True)

    def load(self, filepath: Path) -> None:
        """Load tokenizer from file.

        Args:
            filepath: Path to tokenizer file
        """
        self.tokenizer = HFTokenizer.from_file(str(filepath))
        self._sync_vocab()

    def from_pretrained(self, model_name: str) -> None:
        """Load tokenizer from HuggingFace Hub.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "meta-llama/Llama-2-7b")
        """
        from transformers import AutoTokenizer

        # Load using transformers library
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Convert to HFTokenizer format for consistency
        # Note: This is a simplified approach; full conversion is more complex
        self.tokenizer = HFTokenizer.from_pretrained(model_name)
        self._sync_vocab()
