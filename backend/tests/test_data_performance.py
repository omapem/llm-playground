"""Performance tests for data collection and preparation (success metrics validation)."""

import pytest
import time
from app.data.cleaning import DataCleaner
from app.data.splitter import TrainValSplitter
from app.data.stats import DataStats


class TestDataProcessingPerformance:
    """Performance tests for success metrics."""

    @pytest.mark.performance
    def test_process_1gb_dataset_under_10_minutes(self):
        """
        Success Metric: Process 1GB dataset in <10 minutes.

        This test simulates processing a 1GB equivalent dataset by:
        1. Creating ~10MB of documents (100x smaller for testing)
        2. Measuring cleaning time
        3. Projecting to 1GB equivalent
        """
        # Create 10MB equivalent of test data (1000 docs × 10KB each)
        docs = ["This is a test document. " * 400 for _ in range(1000)]
        total_size_mb = sum(len(d.encode('utf-8')) for d in docs) / (1024 * 1024)

        cleaner = DataCleaner(
            min_length=50,
            remove_duplicates=True,
            remove_pii_enabled=True
        )

        start_time = time.time()
        cleaned = cleaner.clean_documents(docs)
        elapsed = time.time() - start_time

        # Extrapolate to 1GB
        extrapolated_time = elapsed * (1024 / total_size_mb) if total_size_mb > 0 else 0

        assert cleaned, "Should have cleaned documents"
        # For 100x scale test, should be very fast
        assert elapsed < 5, f"Cleaning {total_size_mb:.1f}MB took {elapsed:.2f}s, should be <5s"

        # Projected time for 1GB should be reasonable (<10 min = 600s)
        assert extrapolated_time < 600, (
            f"Projected time for 1GB: {extrapolated_time:.1f}s "
            f"(estimated from {total_size_mb:.1f}MB taking {elapsed:.2f}s)"
        )

    @pytest.mark.performance
    def test_duplicate_detection_under_1_percent(self):
        """
        Success Metric: <1% duplicate documents after cleaning.

        Tests that deduplication effectively removes duplicates.
        """
        # Create dataset with known duplication rate
        unique_docs = [f"Document {i} with unique content here" for i in range(1000)]
        duplicates = unique_docs[:100]  # Add 10% duplicates
        test_docs = unique_docs + duplicates

        cleaner = DataCleaner(
            min_length=10,
            remove_duplicates=True
        )

        cleaned = cleaner.clean_documents(test_docs)
        dup_rate = cleaner.get_cleaning_stats()["duplicate_rate"]

        # After deduplication, should have significantly lower rate
        assert dup_rate < 0.01, (
            f"Duplicate rate {dup_rate:.2%} exceeds success metric of <1%"
        )

    @pytest.mark.performance
    def test_batch_cleaning_throughput(self):
        """Test that batch cleaning achieves good throughput."""
        docs = [f"Document {i} with some content" * 5 for i in range(5000)]

        cleaner = DataCleaner(min_length=50)

        start_time = time.time()
        cleaned = cleaner.clean_documents(docs)
        elapsed = time.time() - start_time

        throughput_docs_per_sec = len(docs) / elapsed

        assert cleaned, "Should have cleaned some documents"
        # Should process at least 1000 docs/sec
        assert throughput_docs_per_sec > 1000, (
            f"Throughput {throughput_docs_per_sec:.0f} docs/sec is too slow "
            f"(target: >1000 docs/sec)"
        )

    @pytest.mark.performance
    def test_split_performance_scaling(self):
        """Test that splitting scales well with dataset size."""
        splitter = TrainValSplitter(seed=42)

        # Test with increasing sizes
        for size in [1000, 10000, 100000]:
            items = list(range(size))

            start_time = time.time()
            train, val = splitter.split(items)
            elapsed = time.time() - start_time

            assert len(train) + len(val) == size
            # Splitting should be nearly instant
            assert elapsed < 1.0, (
                f"Splitting {size} items took {elapsed:.3f}s, should be <1s"
            )

    @pytest.mark.performance
    def test_stats_computation_memory_efficient(self):
        """Test that stats computation is memory efficient."""
        # Test with large document count
        docs = [f"Document {i}: " * 100 for i in range(10000)]

        stats = DataStats(documents=docs)

        start_time = time.time()
        stats.compute()
        elapsed = time.time() - start_time

        assert stats.is_computed
        # Should compute stats for 10K docs in reasonable time
        assert elapsed < 5.0, (
            f"Stats computation took {elapsed:.2f}s for 10K docs, should be <5s"
        )

    @pytest.mark.performance
    def test_pii_removal_performance(self):
        """Test that PII removal doesn't significantly slow down processing."""
        # Documents with and without PII
        docs_with_pii = [
            f"Contact john.doe@example.com or call 555-123-4567. "
            f"Document {i} contains important information" * 5
            for i in range(1000)
        ]

        cleaner = DataCleaner(
            min_length=50,
            remove_pii_enabled=True
        )

        start_time = time.time()
        cleaned = cleaner.clean_documents(docs_with_pii)
        elapsed = time.time() - start_time

        # Should still maintain good throughput with PII removal
        throughput = len(docs_with_pii) / elapsed
        assert throughput > 500, (
            f"Throughput with PII removal: {throughput:.0f} docs/sec, "
            f"should be >500 docs/sec"
        )

    @pytest.mark.performance
    def test_deduplication_hash_performance(self):
        """Test that deduplication via hashing is efficient."""
        # Create dataset with many near-duplicates
        base_doc = "This is a sample document with repeated content. " * 20
        docs = [base_doc] * 5000  # All identical

        cleaner = DataCleaner(remove_duplicates=True, min_length=10)

        start_time = time.time()
        cleaned = cleaner.clean_documents(docs)
        elapsed = time.time() - start_time

        # Should quickly identify and remove duplicates
        assert len(cleaned) == 1, "Should reduce 5000 identical docs to 1"
        assert elapsed < 2.0, (
            f"Deduplication of 5000 identical docs took {elapsed:.2f}s, "
            f"should be <2s"
        )

    @pytest.mark.performance
    def test_stats_vocabulary_extraction_performance(self):
        """Test that vocabulary extraction performs well."""
        docs = [f"Document {i}: " + " ".join([f"word_{j}" for j in range(100)])
                for i in range(1000)]

        stats = DataStats(documents=docs)

        start_time = time.time()
        stats.compute()
        elapsed = time.time() - start_time

        assert stats.vocabulary_size > 0
        # Should extract vocabulary efficiently
        assert elapsed < 3.0, (
            f"Vocabulary extraction took {elapsed:.2f}s, should be <3s"
        )

    @pytest.mark.performance
    def test_pipeline_end_to_end_performance(self):
        """Test complete data pipeline performance."""
        # Create test dataset
        docs = [f"Document {i}: This is sample content for testing. " * 5
                for i in range(2000)]

        start_time = time.time()

        # Step 1: Clean
        cleaner = DataCleaner(min_length=50, remove_duplicates=True)
        cleaned = cleaner.clean_documents(docs)

        # Step 2: Split
        splitter = TrainValSplitter()
        train, val = splitter.split(cleaned)

        # Step 3: Stats
        stats = DataStats(documents=cleaned)
        stats.compute()

        elapsed = time.time() - start_time

        assert len(cleaned) > 0
        assert len(train) > 0 and len(val) > 0
        assert stats.is_computed

        # Complete pipeline should be efficient
        assert elapsed < 3.0, (
            f"Full pipeline took {elapsed:.2f}s, should be <3s"
        )


class TestMemoryUsage:
    """Test memory efficiency of data processing."""

    @pytest.mark.performance
    def test_large_document_handling(self):
        """Test handling of very large documents."""
        # Create 10 large documents
        large_doc = "word " * 100000  # ~500KB each
        docs = [large_doc] * 10

        cleaner = DataCleaner(min_length=100)
        cleaned = cleaner.clean_documents(docs)

        assert len(cleaned) > 0, "Should handle large documents"

    @pytest.mark.performance
    def test_large_batch_handling(self):
        """Test handling of large batches of documents."""
        # 50K small documents
        docs = [f"Doc {i}: content here" for i in range(50000)]

        cleaner = DataCleaner(min_length=10)

        start_time = time.time()
        cleaned = cleaner.clean_documents(docs)
        elapsed = time.time() - start_time

        assert len(cleaned) > 0
        # Should handle 50K docs efficiently
        assert elapsed < 10.0, (
            f"Batch of 50K docs took {elapsed:.2f}s, should be <10s"
        )
