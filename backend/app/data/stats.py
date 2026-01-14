"""Dataset statistics and analysis functionality."""

import hashlib
from typing import List, Optional, Dict, Any
from loguru import logger
from collections import Counter


class DataStats:
    """Compute and track dataset statistics."""

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        compute_language_dist: bool = False,
    ):
        """Initialize DataStats.

        Args:
            documents: List of documents to analyze
            compute_language_dist: Whether to compute language distribution
        """
        self.documents = documents or []
        self.compute_language_dist = compute_language_dist

        # Statistics
        self.num_documents = len(self.documents)
        self.total_characters = 0
        self.total_words = 0
        self.total_tokens = 0
        self.vocabulary_size = 0
        self.min_doc_length = 0
        self.max_doc_length = 0
        self.avg_doc_length = 0.0

        # Advanced statistics
        self.doc_lengths: List[int] = []
        self.token_counts: List[int] = []
        self.vocabulary: set = set()
        self.language_distribution: Dict[str, int] = {}
        self.duplicate_hashes: Dict[str, int] = {}

        self.is_computed = False

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the statistics tracker.

        Args:
            documents: Documents to add
        """
        self.documents.extend(documents)
        self.num_documents = len(self.documents)
        self.is_computed = False

    def compute(self) -> None:
        """Compute all statistics for the documents."""
        if not self.documents:
            logger.warning("No documents to analyze")
            return

        self.doc_lengths = []
        self.token_counts = []
        self.vocabulary = set()
        self.language_distribution = {}
        self.duplicate_hashes = {}

        self.total_characters = 0
        self.total_words = 0
        self.total_tokens = 0

        for doc in self.documents:
            # Character count
            char_count = len(doc)
            self.total_characters += char_count
            self.doc_lengths.append(char_count)

            # Word count (simple split by whitespace)
            words = doc.split()
            word_count = len(words)
            self.total_words += word_count

            # Token approximation (rough: ~1.3 chars per token for English)
            token_count = max(1, len(doc) // 4)  # Conservative estimate
            self.total_tokens += token_count
            self.token_counts.append(token_count)

            # Vocabulary
            for word in words:
                self.vocabulary.add(word.lower())

            # Duplicate detection
            doc_hash = hashlib.md5(doc.encode()).hexdigest()
            self.duplicate_hashes[doc_hash] = self.duplicate_hashes.get(doc_hash, 0) + 1

            # Language distribution (if enabled)
            if self.compute_language_dist:
                from app.data.cleaning import detect_language
                lang = detect_language(doc)
                self.language_distribution[lang] = self.language_distribution.get(lang, 0) + 1

        # Compute aggregates
        self.vocabulary_size = len(self.vocabulary)
        self.min_doc_length = min(self.doc_lengths) if self.doc_lengths else 0
        self.max_doc_length = max(self.doc_lengths) if self.doc_lengths else 0
        self.avg_doc_length = (
            self.total_characters / len(self.documents) if self.documents else 0
        )

        self.is_computed = True
        logger.info(
            f"Computed stats for {self.num_documents} documents: "
            f"avg_doc_length={self.avg_doc_length:.1f}, "
            f"vocab_size={self.vocabulary_size}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of statistics.

        Returns:
            Dictionary with key statistics
        """
        if not self.is_computed:
            self.compute()

        return {
            "num_documents": self.num_documents,
            "total_characters": self.total_characters,
            "total_words": self.total_words,
            "total_tokens": self.total_tokens,
            "vocabulary_size": self.vocabulary_size,
            "avg_doc_length": self.avg_doc_length,
            "min_doc_length": self.min_doc_length,
            "max_doc_length": self.max_doc_length,
            "duplicate_count": sum(1 for count in self.duplicate_hashes.values() if count > 1),
            "duplicate_rate": self.compute_duplicate_rate(),
        }

    def get_detailed_report(self) -> str:
        """Get detailed statistics report as formatted string.

        Returns:
            Formatted statistics report
        """
        if not self.is_computed:
            self.compute()

        report = []
        report.append("=" * 60)
        report.append("DATASET STATISTICS REPORT")
        report.append("=" * 60)
        report.append(f"\nDocuments:")
        report.append(f"  Total: {self.num_documents}")
        report.append(f"  Unique: {self.num_documents - sum(1 for c in self.duplicate_hashes.values() if c > 1)}")
        report.append(f"\nCharacters:")
        report.append(f"  Total: {self.total_characters:,}")
        report.append(f"  Average per doc: {self.avg_doc_length:.1f}")
        report.append(f"  Min: {self.min_doc_length}")
        report.append(f"  Max: {self.max_doc_length}")
        report.append(f"\nWords:")
        report.append(f"  Total: {self.total_words:,}")
        report.append(f"  Average per doc: {self.total_words / max(self.num_documents, 1):.1f}")
        report.append(f"\nTokens (estimated):")
        report.append(f"  Total: {self.total_tokens:,}")
        report.append(f"  Average per doc: {self.total_tokens / max(self.num_documents, 1):.1f}")
        report.append(f"\nVocabulary:")
        report.append(f"  Unique words: {self.vocabulary_size:,}")
        report.append(f"\nQuality:")
        report.append(f"  Duplicate rate: {self.compute_duplicate_rate():.2%}")

        if self.language_distribution:
            report.append(f"\nLanguages:")
            for lang, count in sorted(
                self.language_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                report.append(f"  {lang}: {count} ({count/self.num_documents*100:.1f}%)")

        report.append("=" * 60)
        return "\n".join(report)

    def get_language_distribution(self) -> Dict[str, int]:
        """Get language distribution.

        Returns:
            Dictionary mapping language codes to counts
        """
        if not self.is_computed:
            self.compute()

        return self.language_distribution.copy()

    def get_token_distribution(self) -> Dict[str, Any]:
        """Get token count distribution statistics.

        Returns:
            Dictionary with distribution statistics
        """
        if not self.token_counts:
            return {}

        return {
            "min": min(self.token_counts),
            "max": max(self.token_counts),
            "mean": sum(self.token_counts) / len(self.token_counts),
            "median": sorted(self.token_counts)[len(self.token_counts) // 2],
        }

    def compute_duplicate_rate(self) -> float:
        """Compute the rate of duplicate documents.

        Returns:
            Fraction of documents that are duplicates (0-1)
        """
        if not self.documents:
            return 0.0

        num_duplicates = sum(1 for count in self.duplicate_hashes.values() if count > 1)
        return num_duplicates / len(self.documents)

    def get_percentile(self, metric: str, percentile: float) -> float:
        """Get percentile value for a metric.

        Args:
            metric: Metric name (doc_length, tokens, etc.)
            percentile: Percentile (0-100)

        Returns:
            Percentile value
        """
        if metric == "doc_length":
            data = sorted(self.doc_lengths)
        elif metric == "tokens":
            data = sorted(self.token_counts)
        else:
            return 0.0

        if not data:
            return 0.0

        idx = int(len(data) * (percentile / 100))
        return data[min(idx, len(data) - 1)]

    def export_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary.

        Returns:
            Dictionary with all statistics
        """
        if not self.is_computed:
            self.compute()

        return {
            "num_documents": self.num_documents,
            "total_characters": self.total_characters,
            "total_words": self.total_words,
            "total_tokens": self.total_tokens,
            "vocabulary_size": self.vocabulary_size,
            "min_doc_length": self.min_doc_length,
            "max_doc_length": self.max_doc_length,
            "avg_doc_length": self.avg_doc_length,
            "duplicate_rate": self.compute_duplicate_rate(),
            "token_distribution": self.get_token_distribution(),
            "language_distribution": self.language_distribution,
        }

    @staticmethod
    def compare(stats1: "DataStats", stats2: "DataStats") -> Dict[str, Any]:
        """Compare two dataset statistics.

        Args:
            stats1: First dataset statistics
            stats2: Second dataset statistics

        Returns:
            Dictionary with comparison
        """
        if not stats1.is_computed:
            stats1.compute()
        if not stats2.is_computed:
            stats2.compute()

        return {
            "dataset1_docs": stats1.num_documents,
            "dataset2_docs": stats2.num_documents,
            "dataset1_avg_length": stats1.avg_doc_length,
            "dataset2_avg_length": stats2.avg_doc_length,
            "dataset1_vocab_size": stats1.vocabulary_size,
            "dataset2_vocab_size": stats2.vocabulary_size,
            "dataset1_dup_rate": stats1.compute_duplicate_rate(),
            "dataset2_dup_rate": stats2.compute_duplicate_rate(),
        }
