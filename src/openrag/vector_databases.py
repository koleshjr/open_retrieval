import os
from typing import Optional, List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS, Chroma, Milvus, Qdrant, DocArrayInMemorySearch
from qdrant_client import QdrantClient

class VectorDatabase:
    def __init__(self, vector_store):
        """
        A class for managing vector databases.

        Args:
            vector_store (str): The name of the vector store to use.
            
        """
        self.vector_store = vector_store
        

    def create_index(self, embedding_function: str, index_name: str, docs: Optional[List[Document]], index_dir: Optional[str] = None, **kwargs):
        """
        Creates an index for the given documents using the specified embedding function.

        Args:
            embedding_function (str): The name of the embedding function to use.
            docs (List[Document]): The list of documents to index.
            index_name (str): The name of the index you would like to save the embeddings
            index_dir (Optional[str]): The directory to store the index in.
            **kwargs: Additional arguments specific to the vector store being used.

        Returns:
            The index object.
        """
        if index_dir:
            if os.path.exists(index_dir):
                persist_directory = os.path.join(index_dir, index_name)
            else:
                os.makedirs(index_dir)
                persist_directory = os.path.join(index_dir, index_name)
        else:
            persist_directory = index_name

        def index_exists(index_path: str):
            return os.path.exists(index_path)

        if self.vector_store == 'chroma':

            if index_exists(persist_directory):
                vector_index = Chroma(persist_directory=persist_directory, embedding_function=embedding_function )
            else:
                vector_index = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
                vector_index.persist()
            return vector_index



        elif self.vector_store == 'milvus':
            host = kwargs.get('host','localhost')
            port = kwargs.get('port', 19530)
            if index_exists(persist_directory):
                vector_index = Milvus(embedding_function=embedding_function, connection_args = {"host": host, "port": port, "collection_name": index_name})
            else:
                vector_index = Milvus.from_documents(docs, embedding_function, collection_name=index_name,
                                                    connection_args={'host': host,
                                                                    'port': port})
            return vector_index

        elif self.vector_store == 'qdrant':
            qdrant_environment = kwargs.get('environment', 'disk')
            if qdrant_environment == 'memory':
                vector_index = Qdrant.from_documents(docs, embedding_function, collection_name=index_name,
                                                      location=":memory:")
                return vector_index
            elif qdrant_environment == 'disk':
                if index_exists(os.path.join(index_dir, index_name)):
                    client = QdrantClient(path = index_dir)
                    Qdrant(client = client, collection_name = index_name, embeddings=embedding_function)
                else:
                    vector_index = Qdrant.from_documents(docs, embedding_function, collection_name=index_name,
                                                      path=index_dir)
                return vector_index

            else:
                raise ValueError(
                    'Invalid environment value: Expecting one of memory, disk, on_premise or cloud')

        elif self.vector_store == 'array':
            vector_index = DocArrayInMemorySearch.from_documents(docs, embedding_function, index_name=persist_directory)
            return vector_index

        elif self.vector_store == 'faiss':
            if index_exists(os.path.join(index_dir, index_name)):
                vector_index = FAISS.load_local(persist_directory, embeddings=embedding_function, allow_dangerous_deserialization=True)
            else:
                vector_index = FAISS.from_documents(docs, embedding_function)
                vector_index.save_local(persist_directory)
            return vector_index

        else:
            raise ValueError(
                'Invalid vector_store value: Expecting one of chroma, milvus, qdrant, faiss or array')
        
