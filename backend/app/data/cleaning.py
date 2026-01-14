"""Data cleaning and quality filtering functionality."""

import re
from typing import List, Optional, Dict, Any
from loguru import logger
import hashlib


def remove_pii(text: str, placeholder: str = "[PII]") -> str:
    """Remove personally identifiable information from text.

    Args:
        text: Input text
        placeholder: String to replace PII with

    Returns:
        Text with PII removed
    """
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]", text)

    # Remove phone numbers (various formats)
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                  "[PHONE]", text)

    # Remove SSN format (XXX-XX-XXXX)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', "[SSN]", text)

    # Remove credit card numbers
    text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', "[CARD]", text)

    # Remove IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', "[IP]", text)

    return text


def detect_language(text: str) -> str:
    """Detect the language of text using langdetect.

    Args:
        text: Text to detect language of

    Returns:
        Language code (e.g., 'en', 'es', 'fr') or 'unknown'
    """
    if len(text.strip()) < 10:
        return "unknown"

    try:
        # Try to import langdetect, fall back to simple heuristic
        try:
            from langdetect import detect
            return detect(text)
        except ImportError:
            # Simple heuristic-based language detection
            text_lower = text.lower()

            # Common English words
            if any(word in text_lower for word in ["the", "is", "and", "to", "of", "a"]):
                return "en"

            # Common Spanish words
            if any(word in text_lower for word in ["el", "la", "es", "de", "y", "en"]):
                return "es"

            # Common French words
            if any(word in text_lower for word in ["le", "la", "est", "de", "et", "un"]):
                return "fr"

            # German words
            if any(word in text_lower for word in ["der", "die", "und", "ist", "ein"]):
                return "de"

            return "unknown"
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
        return "unknown"


def deduplicate(texts: List[str]) -> List[str]:
    """Remove duplicate texts from a list.

    Args:
        texts: List of text documents

    Returns:
        List with duplicates removed (preserves order)
    """
    seen = set()
    unique = []

    for text in texts:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(text)

    return unique


class DataCleaner:
    """Clean and filter text data."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: Optional[int] = None,
        remove_duplicates: bool = True,
        remove_pii_enabled: bool = True,
        allowed_languages: Optional[List[str]] = None,
        filter_quality: bool = False,
    ):
        """Initialize DataCleaner.

        Args:
            min_length: Minimum length of text to keep
            max_length: Maximum length of text to keep
            remove_duplicates: Whether to remove duplicate documents
            remove_pii_enabled: Whether to remove PII
            allowed_languages: List of allowed language codes
            filter_quality: Whether to filter low-quality text
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.remove_pii_enabled = remove_pii_enabled
        self.allowed_languages = allowed_languages or ["en"]
        self.filter_quality = filter_quality

        self.stats = {
            "documents_processed": 0,
            "documents_kept": 0,
            "documents_filtered": 0,
            "total_chars_before": 0,
            "total_chars_after": 0,
        }

    def clean_text(self, text: str) -> str:
        """Clean a single text document.

        Args:
            text: Text to clean

        Returns:
            Cleaned text (may be empty if filtered out)
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Check minimum length before processing
        if len(text) < self.min_length:
            return ""

        # Check maximum length
        if self.max_length and len(text) > self.max_length:
            return ""

        # Remove PII if enabled
        if self.remove_pii_enabled:
            text = remove_pii(text)

        # Normalize whitespace (collapse multiple spaces)
        text = re.sub(r'\s+', ' ', text).strip()

        # Language filtering
        if self.allowed_languages and self.filter_quality:
            lang = detect_language(text)
            if lang not in self.allowed_languages and lang != "unknown":
                return ""

        # Final length check after cleaning
        if len(text) < self.min_length:
            return ""

        return text

    def clean_documents(self, texts: List[str]) -> List[str]:
        """Clean a batch of documents.

        Args:
            texts: List of texts to clean

        Returns:
            List of cleaned texts
        """
        cleaned = []
        seen_hashes = set()

        for text in texts:
            self.stats["documents_processed"] += 1
            self.stats["total_chars_before"] += len(text)

            cleaned_text = self.clean_text(text)

            if cleaned_text:
                # Check for duplicates
                text_hash = hashlib.md5(cleaned_text.encode()).hexdigest()

                if self.remove_duplicates:
                    if text_hash not in seen_hashes:
                        cleaned.append(cleaned_text)
                        seen_hashes.add(text_hash)
                        self.stats["documents_kept"] += 1
                        self.stats["total_chars_after"] += len(cleaned_text)
                    else:
                        self.stats["documents_filtered"] += 1
                else:
                    cleaned.append(cleaned_text)
                    self.stats["documents_kept"] += 1
                    self.stats["total_chars_after"] += len(cleaned_text)
            else:
                self.stats["documents_filtered"] += 1

        logger.info(
            f"Cleaned {self.stats['documents_processed']} documents: "
            f"kept {self.stats['documents_kept']}, "
            f"filtered {self.stats['documents_filtered']}"
        )

        return cleaned

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get statistics about the cleaning process.

        Returns:
            Dictionary with cleaning statistics
        """
        return {
            **self.stats,
            "duplicate_rate": (
                (self.stats["documents_processed"] - self.stats["documents_kept"])
                / max(self.stats["documents_processed"], 1) * 100
            ),
            "size_reduction_ratio": (
                self.stats["total_chars_after"] / max(self.stats["total_chars_before"], 1)
            ),
        }

    def reset_stats(self) -> None:
        """Reset cleaning statistics."""
        self.stats = {
            "documents_processed": 0,
            "documents_kept": 0,
            "documents_filtered": 0,
            "total_chars_before": 0,
            "total_chars_after": 0,
        }
