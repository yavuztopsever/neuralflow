import pytest
from hypothesis import given, strategies as st
import numpy as np
from src.models.data_models import TextEmbedding, SearchResult
from src.storage.vector_store import VectorStore

@pytest.mark.unit
class TestDataValidation:
    @pytest.fixture
    def vector_store(self):
        store = VectorStore()
        yield store
        store.clear()

    @given(
        text=st.text(min_size=1, max_size=1000),
        vector=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=768, max_size=768)
    )
    def test_text_embedding_validation(self, text, vector):
        # Test valid data
        embedding = TextEmbedding(text=text, vector=vector)
        assert embedding.text == text
        assert len(embedding.vector) == 768
        assert all(-1.0 <= v <= 1.0 for v in embedding.vector)

        # Test invalid data
        with pytest.raises(ValueError):
            TextEmbedding(text="", vector=vector)  # Empty text

        with pytest.raises(ValueError):
            TextEmbedding(text=text, vector=vector + [1.0])  # Wrong vector size

        with pytest.raises(ValueError):
            TextEmbedding(text=text, vector=[2.0] * 768)  # Vector values out of range

    @given(
        text=st.text(min_size=1, max_size=1000),
        score=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_search_result_validation(self, text, score):
        # Test valid data
        result = SearchResult(text=text, score=score)
        assert result.text == text
        assert 0.0 <= result.score <= 1.0

        # Test invalid data
        with pytest.raises(ValueError):
            SearchResult(text="", score=score)  # Empty text

        with pytest.raises(ValueError):
            SearchResult(text=text, score=1.5)  # Score out of range

    @given(
        texts=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=100),
        vectors=st.lists(
            st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=768, max_size=768),
            min_size=1,
            max_size=100
        )
    )
    def test_bulk_validation(self, vector_store, texts, vectors):
        # Ensure lists have same length
        if len(texts) != len(vectors):
            vectors = vectors[:len(texts)]
            texts = texts[:len(vectors)]

        # Test valid bulk data
        vector_store.add_texts(texts, vectors)
        assert vector_store.count() == len(texts)

        # Test invalid bulk data
        with pytest.raises(ValueError):
            vector_store.add_texts(texts, vectors + [[1.0] * 768])  # Mismatched lengths

        with pytest.raises(ValueError):
            vector_store.add_texts(texts + [""], vectors)  # Empty text in list

    @given(
        query_vector=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=768, max_size=768),
        k=st.integers(min_value=1, max_value=100)
    )
    def test_search_validation(self, vector_store, query_vector, k):
        # Setup test data
        num_vectors = max(k + 1, 10)
        vectors = [np.random.randn(768).tolist() for _ in range(num_vectors)]
        texts = [f"Test text {i}" for i in range(num_vectors)]
        vector_store.add_texts(texts, vectors)

        # Test valid search
        results = vector_store.similarity_search(query_vector, k=k)
        assert len(results) == k
        assert all(0.0 <= result.score <= 1.0 for result in results)

        # Test invalid search
        with pytest.raises(ValueError):
            vector_store.similarity_search(query_vector, k=0)  # Invalid k value

        with pytest.raises(ValueError):
            vector_store.similarity_search(query_vector + [1.0], k=k)  # Wrong vector size 