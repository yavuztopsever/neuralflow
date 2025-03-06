import pytest
import numpy as np
from src.core.workflow import WorkflowManager
from src.storage.vector_store import VectorStore
from src.models.data_models import TextEmbedding, SearchResult
from src.utils.text_processor import TextProcessor

@pytest.mark.e2e
class TestCompleteWorkflow:
    @pytest.fixture
    def workflow(self):
        workflow = WorkflowManager()
        yield workflow
        workflow.cleanup()

    @pytest.fixture
    def sample_data(self):
        return [
            "The quick brown fox jumps over the lazy dog.",
            "A quick brown dog runs in the park.",
            "The lazy fox sleeps under the tree.",
            "A dog and a fox play in the park.",
            "The quick fox runs through the park."
        ]

    def test_text_processing_workflow(self, workflow, sample_data):
        # Test text processing
        processed_texts = []
        for text in sample_data:
            processed = workflow.process_text(text)
            processed_texts.append(processed)
            assert isinstance(processed, str)
            assert len(processed) > 0

    def test_embedding_generation_workflow(self, workflow, sample_data):
        # Test embedding generation
        embeddings = []
        for text in sample_data:
            embedding = workflow.generate_embedding(text)
            embeddings.append(embedding)
            assert isinstance(embedding, list)
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    def test_vector_store_workflow(self, workflow, sample_data):
        # Test vector store operations
        # Add texts to vector store
        for text in sample_data:
            workflow.add_to_vector_store(text)

        # Verify storage
        assert workflow.vector_store.count() == len(sample_data)

        # Test search
        query = "quick fox"
        results = workflow.search_similar(query, k=3)
        assert len(results) == 3
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(0.0 <= result.score <= 1.0 for result in results)

    def test_complete_pipeline(self, workflow, sample_data):
        # Test the complete pipeline from text input to search results
        # 1. Process and store texts
        for text in sample_data:
            workflow.process_and_store(text)

        # 2. Verify storage
        assert workflow.vector_store.count() == len(sample_data)

        # 3. Test search with different queries
        queries = [
            "quick fox",
            "lazy dog",
            "park activities"
        ]

        for query in queries:
            results = workflow.search_similar(query, k=2)
            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert all(0.0 <= result.score <= 1.0 for result in results)

    def test_error_handling(self, workflow):
        # Test error handling in the workflow
        with pytest.raises(ValueError):
            workflow.process_text("")  # Empty text

        with pytest.raises(ValueError):
            workflow.search_similar("", k=0)  # Invalid k value

        with pytest.raises(ValueError):
            workflow.search_similar("test", k=-1)  # Negative k value

    def test_concurrent_operations(self, workflow, sample_data):
        import concurrent.futures
        import random

        def process_and_search(text):
            workflow.process_and_store(text)
            query = random.choice(sample_data)
            return workflow.search_similar(query, k=2)

        # Test concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_and_search, text) for text in sample_data]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == len(sample_data)
        assert all(len(result) == 2 for result in results)

    def test_memory_management(self, workflow, sample_data):
        import psutil
        process = psutil.Process()

        # Measure initial memory
        initial_memory = process.memory_info().rss

        # Process and store texts
        for text in sample_data:
            workflow.process_and_store(text)

        # Measure final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Assert reasonable memory usage
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase 