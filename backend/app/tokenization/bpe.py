"""Byte Pair Encoding (BPE) tokenizer implementation.

This module provides an educational implementation of Byte Pair Encoding,
the tokenization algorithm used by GPT-2 and many modern language models.

Key concepts:
- Start with character-level vocabulary
- Iteratively merge most frequent adjacent pairs
- Results in subword tokens that balance character coverage and vocabulary size
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .base import BaseTokenizer, TokenizationResult


class BPETokenizer(BaseTokenizer):
    """Educational Byte Pair Encoding tokenizer implementation.

    This implementation follows the GPT-2 style BPE algorithm:
    1. Initialize with character-level vocabulary
    2. Count pair frequencies in the corpus
    3. Merge most frequent pair
    4. Repeat until reaching target vocabulary size

    For learning purposes, this implementation prioritizes clarity over
    production performance. For real usage, see HuggingFaceTokenizerWrapper.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        unk_token: str = "[UNK]",
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size
            unk_token: Token for unknown words
            special_tokens: List of special tokens to reserve
        """
        super().__init__(vocab_size, unk_token)

        self.special_tokens = special_tokens or []
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.cache: Dict[str, str] = {}

        # Initialize with special tokens and ASCII characters
        self._initialize_vocab()

    def _initialize_vocab(self) -> None:
        """Initialize vocabulary with special tokens and characters."""
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token

        # Add ASCII characters (printable range)
        vocab_idx = len(self.special_tokens)
        for i in range(256):
            byte_token = f"<|byte_{i}|>" if i < 32 or i == 127 else chr(i)
            self.vocab[byte_token] = vocab_idx
            self.reverse_vocab[vocab_idx] = byte_token
            vocab_idx += 1

    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        """Train BPE tokenizer on a corpus.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size (overrides init value)
            show_progress: Whether to show progress during training
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        # Count character n-grams in the corpus
        word_freqs = self._get_word_frequencies(texts)

        # Convert words to character sequences with frequency info
        word_splits = {
            " ".join(word) + " </w>": freq for word, freq in word_freqs.items()
        }

        # Merge pairs until reaching vocabulary size
        current_vocab_size = self.get_vocab_size()
        merge_count = 0

        while current_vocab_size < self.vocab_size:
            # Find most common adjacent pair
            pair_freqs = self._count_pair_frequencies(word_splits)

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Merge the best pair in all words
            word_splits = self._merge_pair(word_splits, best_pair)

            # Record the merge and add new token to vocabulary
            self.merges.append(best_pair)
            new_token = "".join(best_pair)
            self.vocab[new_token] = current_vocab_size
            self.reverse_vocab[current_vocab_size] = new_token
            self.bpe_ranks[best_pair] = merge_count

            current_vocab_size += 1
            merge_count += 1

            if show_progress and merge_count % 100 == 0:
                print(f"Merged {merge_count} pairs, vocab size: {current_vocab_size}")

    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Extract word frequencies from texts.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary mapping words to frequencies
        """
        word_freqs = Counter()
        for text in texts:
            # Simple tokenization on whitespace
            words = text.split()
            word_freqs.update(words)
        return dict(word_freqs)

    def _count_pair_frequencies(
        self, word_splits: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        """Count frequencies of adjacent token pairs.

        Args:
            word_splits: Dictionary mapping word splits to frequencies

        Returns:
            Dictionary mapping pairs to their total frequency
        """
        pair_freqs = defaultdict(int)

        for word, freq in word_splits.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq

        return dict(pair_freqs)

    def _merge_pair(
        self, word_splits: Dict[str, int], pair: Tuple[str, str]
    ) -> Dict[str, int]:
        """Merge a specific pair in all words.

        Args:
            word_splits: Current word splits dictionary
            pair: Pair of tokens to merge

        Returns:
            Updated word splits with merged pairs
        """
        new_word_splits = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word, freq in word_splits.items():
            new_word = word.replace(bigram, replacement)
            new_word_splits[new_word] = freq

        return new_word_splits

    def encode(self, text: str) -> TokenizationResult:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            TokenizationResult with tokens, token_ids, and offsets
        """
        # Preprocess and split into words
        words = text.split()
        tokens = []
        offsets = []
        current_offset = 0

        for word in words:
            # Convert word to characters
            word_tokens = list(word) + ["</w>"]

            # Apply BPE merges
            word_tokens = self._apply_bpe(word_tokens)

            # Track offsets
            for token in word_tokens:
                tokens.append(token)
                if token != "</w>":
                    offsets.append((current_offset, current_offset + len(token)))
                    current_offset += len(token)

            current_offset += 1  # Space

        # Convert tokens to IDs
        token_ids = [self.token_to_id(token) for token in tokens]

        return TokenizationResult(
            tokens=tokens,
            token_ids=token_ids,
            offsets=offsets,
        )

    def _apply_bpe(self, word_tokens: List[str]) -> List[str]:
        """Apply learned BPE merges to a word.

        Args:
            word_tokens: Word as character list

        Returns:
            Word as BPE token list
        """
        # Cache for efficiency
        word_str = " ".join(word_tokens)
        if word_str in self.cache:
            return self.cache[word_str].split()

        # Apply merges in order
        tokens = word_tokens
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append("".join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Cache result
        self.cache[word_str] = " ".join(tokens)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = [self.id_to_token(token_id) for token_id in token_ids]

        # Remove </w> markers and join
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def save(self, filepath: Path) -> None:
        """Save tokenizer to JSON file.

        Args:
            filepath: Path to save tokenizer
        """
        state = {
            "vocab": self.vocab,
            "merges": [(a, b) for a, b in self.merges],
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: Path) -> None:
        """Load tokenizer from JSON file.

        Args:
            filepath: Path to tokenizer file
        """
        with open(filepath, "r") as f:
            state = json.load(f)

        self.vocab = state["vocab"]
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in state["merges"]]
        self.bpe_ranks = {m: i for i, m in enumerate(self.merges)}
        self.special_tokens = state["special_tokens"]
        self.vocab_size = state["vocab_size"]
