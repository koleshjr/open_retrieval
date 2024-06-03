import os
import sys
import pytest
# from langchain_community.vectorstores import Milvus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.open_retrieval.document_loaders import DocumentLoader
from src.open_retrieval.embedding_providers import EmbeddingProvider
from src.open_retrieval.text_splitters import TextSplitter
from src.open_retrieval.vector_databases import VectorDatabase  
from src.open_retrieval.retrievers import Retriever
from rerankers import Reranker
class TestVectorDatabase:
    document_loader = DocumentLoader()
    text_splitter = TextSplitter(splitter='recursive')
    embedding_provider = EmbeddingProvider('huggingface')
    def faiss_vector_database(self):
        return VectorDatabase(vector_store='faiss')
    @pytest.fixture
    def test_ranked_retrieval(self, faiss_vector_database):
        index_dir='tests/index/'
        index_name = 'test_faiss'
        ranking_model = "colbert"
        ranker = Reranker(ranking_model, verbose=0)
        embedding_function =  self.embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        vector_index = faiss_vector_database.create_index(embedding_function=embedding_function, index_name=index_name,index_dir=index_dir)
        retriever = Retriever(vector_index=vector_index, ranker = ranker)
        results = retriever.ranked_retrieval( query='What are the general obligations for WHO', top_k=15, ranked_top_k=5 )
        assert len(results) == 5

    @pytest.fixture
    def test_naive_retrieval(self, faiss_vector_database):
        index_dir='tests/index/'
        index_name = 'test_faiss'
        embedding_function =  self.embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        vector_index = faiss_vector_database.create_index(embedding_function=embedding_function, index_name=index_name,index_dir=index_dir)
        retriever = Retriever(vector_index=vector_index)
        results = retriever.naive_retrieval( query='What are MEASURES RELATING TO THE REDUCTION OF THE SUPPLY OF TOBACCO', top_k=5 )
        assert len(results) == 5

