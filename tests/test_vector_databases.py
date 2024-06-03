import os
import sys
import pytest
# from langchain_community.vectorstores import Milvus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.openrag.document_loaders import DocumentLoader
from src.openrag.embedding_providers import EmbeddingProvider
from src.openrag.text_splitters import TextSplitter
from src.openrag.vector_databases import VectorDatabase  
class TestVectorDatabase:
    document_loader = DocumentLoader()
    text_splitter = TextSplitter(splitter='recursive')
    embedding_provider = EmbeddingProvider('huggingface')
    
    @pytest.fixture
    def chroma_vector_database(self):
        return VectorDatabase(vector_store='chroma')
    
    def test_create_index_chroma(self, chroma_vector_database):
        index_dir='tests/index/'
        index_name = 'test_chroma'
        file_path = "data/rag_data/1_WHO_FCTC.pdf"
        embedding_function =  self.embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        data = self.document_loader.load(file_path=file_path)
        docs = self.text_splitter.split(data)
        chroma_vector_database.create_index(embedding_function=embedding_function,docs=docs, index_name=index_name,index_dir=index_dir)
        assert os.path.exists(os.path.join(index_dir, "test_chroma"))

    @pytest.fixture
    def faiss_vector_database(self):
        return VectorDatabase(vector_store='faiss')
    
    def test_create_index_faiss(self, faiss_vector_database):
        index_dir='tests/index/'
        index_name = 'test_faiss'
        file_path = "data/rag_data/1_WHO_FCTC.pdf"
        embedding_function =  self.embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        data = self.document_loader.load(file_path=file_path)
        docs = self.text_splitter.split(data)
        faiss_vector_database.create_index(embedding_function=embedding_function,docs=docs, index_name=index_name,index_dir=index_dir)
        assert "test_faiss" in os.listdir(index_dir)
        assert os.path.exists(os.path.join(index_dir, "test_faiss"))

    @pytest.fixture
    def qdrant_vector_database(self):
        return VectorDatabase(vector_store='qdrant')
    
    def test_create_index_qdrant(self, qdrant_vector_database):
        index_dir='tests/index/'
        index_name = 'test_qdrant'
        file_path = "data/rag_data/1_WHO_FCTC.pdf"
        embedding_function =  self.embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        data = self.document_loader.load(file_path=file_path)
        docs = self.text_splitter.split(data)
        qdrant_vector_database.create_index(embedding_function=embedding_function,docs=docs, index_name=index_name,index_dir=index_dir)
        assert os.path.exists(os.path.join(index_dir, "collection", "test_qdrant"))

    # #this milvus test should be done locally with a linux system or wsl in windows
    # @pytest.fixture
    # def milvus_vector_database(self):
    #     return VectorDatabase(vector_store='milvus', index_name='test_milvus')
    
    # def test_create_index_milvus(self, milvus_vector_database):
    #     index_dir='tests/index/'
    #     index_name = 'test_milvus'
    #     file_path = "data/rag_data/1_WHO_FCTC.pdf"
    #     embedding_function =  self.embedding_provider.get_embedding_function()
    #     data = self.document_loader.load(file_path=file_path)
    #     docs = self.text_splitter.split(data)
    #     milvus_vector_database.create_index(embedding_function=embedding_function,docs=docs, index_name=index_name,index_dir=index_dir)
        
    #     assert Milvus(embedding_function, connection_args = {"host": Settings.get_milvus_host(), "port": Settings.get_milvus_port() }, collection_name="test_milvus") is not None

    @pytest.fixture
    def array_vector_database(self):
        return VectorDatabase(vector_store='array')
    
    def test_create_index_array(self, array_vector_database):
        index_dir='tests/index/'
        index_name='test_array'
        file_path = "data/rag_data/1_WHO_FCTC.pdf"
        embedding_function =  self.embedding_provider.get_embedding_function(model_name = "BAAI/bge-base-en-v1.5")
        data = self.document_loader.load(file_path=file_path)
        docs = self.text_splitter.split(data)
        assert array_vector_database.create_index(embedding_function=embedding_function,docs=docs, index_name=index_name,index_dir=index_dir) is not None
        