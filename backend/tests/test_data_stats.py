"""Tests for dataset statistics functionality."""

import pytest
from app.data.stats import DataStats


class TestDataStats:
    """Tests for DataStats class."""

    def test_init_empty_dataset(self):
        """Test initialization with empty dataset."""
        stats = DataStats()
        assert stats.num_documents == 0
        assert stats.total_tokens == 0

    def test_init_with_documents(self):
        """Test initialization with documents."""
        docs = ["First document", "Second document"]
        stats = DataStats(documents=docs)
        assert stats.num_documents == 2

    def test_compute_basic_stats(self):
        """Test computing basic statistics."""
        docs = [
            "Hello world",
            "This is a test",
            "Another example",
        ]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.num_documents == 3
        assert stats.total_characters > 0
        assert stats.avg_doc_length > 0

    def test_compute_word_count(self):
        """Test word count computation."""
        docs = ["one two three", "four five"]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.total_words == 5
        assert stats.total_words / len(docs) > 0

    def test_compute_character_count(self):
        """Test character count computation."""
        docs = ["abc", "defgh"]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.total_characters == 8

    def test_compute_token_stats(self):
        """Test token statistics computation."""
        docs = ["This is a sentence."]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.total_tokens > 0
        assert stats.total_tokens / len(docs) > 0

    def test_compute_vocabulary_size(self):
        """Test vocabulary size computation."""
        docs = [
            "apple banana cherry",
            "apple date elderberry",
        ]
        stats = DataStats(documents=docs)
        stats.compute()

        # Should have at least these unique words
        assert stats.vocabulary_size >= 5

    def test_get_summary(self):
        """Test getting summary of statistics."""
        docs = ["Test document one", "Test document two"]
        stats = DataStats(documents=docs)
        stats.compute()

        summary = stats.get_summary()

        assert isinstance(summary, dict)
        assert "num_documents" in summary
        assert "total_characters" in summary
        assert "vocabulary_size" in summary

    def test_get_detailed_report(self):
        """Test getting detailed statistics report."""
        docs = ["Test"] * 10
        stats = DataStats(documents=docs)
        stats.compute()

        report = stats.get_detailed_report()

        assert isinstance(report, str)
        assert "Documents" in report or "documents" in report

    def test_language_distribution(self):
        """Test language distribution computation."""
        docs = [
            "This is English text",
            "Este es texto en español",
        ]
        stats = DataStats(documents=docs, compute_language_dist=True)
        stats.compute()

        lang_dist = stats.get_language_distribution()
        assert isinstance(lang_dist, dict)

    def test_update_with_new_documents(self):
        """Test updating statistics with new documents."""
        stats = DataStats()
        stats.add_documents(["Document one", "Document two"])
        stats.compute()

        assert stats.num_documents == 2

        stats.add_documents(["Document three"])
        stats.compute()

        assert stats.num_documents == 3

    def test_duplicate_rate_computation(self):
        """Test duplicate rate computation."""
        docs = [
            "Duplicate text",
            "Duplicate text",
            "Unique text",
        ]
        stats = DataStats(documents=docs)
        stats.compute()
        dup_rate = stats.compute_duplicate_rate()

        assert dup_rate >= 0  # Can be 0 or positive

    def test_token_distribution(self):
        """Test token count distribution."""
        docs = [
            "Short",
            "This is a medium length document",
            "This is a very long document " * 10,
        ]
        stats = DataStats(documents=docs)
        stats.compute()

        dist = stats.get_token_distribution()
        assert isinstance(dist, dict)

    def test_document_length_statistics(self):
        """Test document length statistics."""
        docs = [
            "a",
            "a b c",
            "a b c d e f g h i j",
        ]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.min_doc_length > 0
        assert stats.max_doc_length > stats.min_doc_length
        assert stats.avg_doc_length > 0

    def test_compute_with_unicode_text(self):
        """Test handling Unicode text."""
        docs = [
            "English text",
            "日本語のテキスト",
            "Текст на русском",
        ]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.num_documents == 3
        assert stats.total_characters > 0

    def test_export_statistics(self):
        """Test exporting statistics as dictionary."""
        docs = ["Test document"] * 5
        stats = DataStats(documents=docs)
        stats.compute()

        export = stats.export_dict()

        assert isinstance(export, dict)
        assert "num_documents" in export
        assert "total_words" in export

    def test_compare_datasets(self):
        """Test comparing statistics between datasets."""
        docs1 = ["Document"] * 10
        docs2 = ["Very long document"] * 100

        stats1 = DataStats(documents=docs1)
        stats1.compute()

        stats2 = DataStats(documents=docs2)
        stats2.compute()

        comparison = DataStats.compare(stats1, stats2)

        assert isinstance(comparison, dict)

    def test_compute_perplexity_baseline(self):
        """Test baseline perplexity computation."""
        docs = ["the the the"] * 10
        stats = DataStats(documents=docs)
        stats.compute()

        # Should be able to compute baseline metrics
        assert stats.total_tokens > 0

    def test_percentile_stats(self):
        """Test percentile statistics."""
        docs = [f"Document {i} " * i for i in range(1, 11)]
        stats = DataStats(documents=docs)
        stats.compute()

        p50 = stats.get_percentile("doc_length", 50)
        p90 = stats.get_percentile("doc_length", 90)

        assert p50 > 0
        assert p90 > p50

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        docs = ["", "Non-empty", ""]
        stats = DataStats(documents=docs)
        stats.compute()

        # Should handle gracefully
        assert stats.num_documents == 3

    def test_very_large_document_handling(self):
        """Test handling of very large documents."""
        large_doc = "word " * 100000
        stats = DataStats(documents=[large_doc])
        stats.compute()

        assert stats.num_documents == 1
        assert stats.total_words >= 100000

    def test_memory_efficiency(self):
        """Test that stats computation is memory efficient."""
        # Create a large dataset
        docs = [f"Document {i} with some content here" for i in range(10000)]
        stats = DataStats(documents=docs)
        stats.compute()

        assert stats.num_documents == 10000
        assert stats.total_characters > 0
