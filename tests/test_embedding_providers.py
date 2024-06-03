import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.open_retrieval.embedding_providers import EmbeddingProvider

class TestEmbeddingProvider:
    @pytest.fixture
    def huggingface_embedding_provider(self):
        return EmbeddingProvider(embedding_provider="huggingface")

    def test_get_embedding_function_huggingface(self, huggingface_embedding_provider):
        embedding_function = huggingface_embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        assert embedding_function is not None 

    @pytest.fixture
    def qdrant_embedding_provider(self):
        return EmbeddingProvider(embedding_provider="fastembed")

    def test_get_embedding_function_qdrant(self, qdrant_embedding_provider):
        embedding_function = qdrant_embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        assert embedding_function is not None 

    @pytest.fixture
    def ollama_embedding_provider(self):
        return EmbeddingProvider(embedding_provider="ollama")

    def test_get_embedding_function_ollama(self, ollama_embedding_provider):
        embedding_function = ollama_embedding_provider.get_embedding_function()
        assert embedding_function is not None 
