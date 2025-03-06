import pytest
import psutil
import time
from typing import List
import numpy as np
from src.storage.vector_store import VectorStore

@pytest.mark.performance
class TestVectorStorePerformance:
    @pytest.fixture
    def vector_store(self):
        store = VectorStore()
        yield store
        store.clear()

    def test_bulk_insertion_performance(self, vector_store):
        # Generate test data
        num_vectors = 1000
        vector_dim = 768
        vectors = [np.random.randn(vector_dim).tolist() for _ in range(num_vectors)]
        texts = [f"Test text {i}" for i in range(num_vectors)]

        # Measure insertion time
        start_time = time.time()
        vector_store.add_texts(texts, vectors)
        insertion_time = time.time() - start_time

        # Assert reasonable performance
        assert insertion_time < 5.0  # Should complete within 5 seconds

    def test_memory_usage(self, vector_store):
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Add some data
        num_vectors = 100
        vector_dim = 768
        vectors = [np.random.randn(vector_dim).tolist() for _ in range(num_vectors)]
        texts = [f"Test text {i}" for i in range(num_vectors)]
        vector_store.add_texts(texts, vectors)

        # Measure memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Assert reasonable memory usage (less than 100MB for 100 vectors)
        assert memory_increase < 100 * 1024 * 1024  # 100MB in bytes

    def test_search_performance(self, vector_store):
        # Setup test data
        num_vectors = 1000
        vector_dim = 768
        vectors = [np.random.randn(vector_dim).tolist() for _ in range(num_vectors)]
        texts = [f"Test text {i}" for i in range(num_vectors)]
        vector_store.add_texts(texts, vectors)

        # Test search performance
        query_vector = np.random.randn(vector_dim).tolist()
        k = 10

        # Measure search time
        start_time = time.time()
        results = vector_store.similarity_search(query_vector, k=k)
        search_time = time.time() - start_time

        # Assert reasonable search performance
        assert search_time < 0.1  # Should complete within 100ms
        assert len(results) == k

    def test_concurrent_access(self, vector_store):
        import concurrent.futures
        import random

        def worker():
            vector_dim = 768
            query_vector = np.random.randn(vector_dim).tolist()
            return vector_store.similarity_search(query_vector, k=5)

        # Setup test data
        num_vectors = 1000
        vector_dim = 768
        vectors = [np.random.randn(vector_dim).tolist() for _ in range(num_vectors)]
        texts = [f"Test text {i}" for i in range(num_vectors)]
        vector_store.add_texts(texts, vectors)

        # Test concurrent access
        num_threads = 10
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time

        # Assert reasonable concurrent performance
        assert total_time < 1.0  # Should complete within 1 second
        assert len(results) == num_threads 