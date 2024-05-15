import os
from typing import Optional, List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS, Chroma, Milvus, Qdrant, DocArrayInMemorySearch

class VectorDatabase:
    def __init__(self, vector_store, index_name):
        """
        A class for managing vector databases.

        Args:
            vector_store (str): The name of the vector store to use.
            index_name (str): The name of the index to use.
        """
        self.vector_store = vector_store
        self.index_name = index_name

    def create_index(self, embedding_function: str, docs: List[Document], index_dir: Optional[str] = None, **kwargs):
        """
        Creates an index for the given documents using the specified embedding function.

        Args:
            embedding_function (str): The name of the embedding function to use.
            docs (List[Document]): The list of documents to index.
            index_dir (Optional[str]): The directory to store the index in. If not specified, the index will be stored in memory.
            **kwargs: Additional arguments specific to the vector store being used.

        Returns:
            The index object.
        """
        if index_dir:
            if os.path.exists(index_dir):
                persist_directory = os.path.join(index_dir, self.index_name)
            else:
                os.makedirs(index_dir)
                persist_directory = os.path.join(index_dir, self.index_name)
        else:
            persist_directory = self.index_name

        if self.vector_store == 'chroma':
            vector_index = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
            return vector_index.persist()



        elif self.vector_store == 'milvus':
            host = kwargs.get('host','localhost')
            port = kwargs.get('port', 19530)
            vector_index = Milvus.from_documents(docs, embedding_function, collection_name=self.index_name,
                                                  connection_args={'host': host,
                                                                   'port': port})

            return vector_index

        elif self.vector_store == 'qdrant':
            qdrant_environment = kwargs.get('environment', 'disk')
            if qdrant_environment == 'memory':
                vector_index = Qdrant.from_documents(docs, embedding_function, collection_name=self.index_name,
                                                      location=":memory:")
            elif qdrant_environment == 'disk':
                vector_index = Qdrant.from_documents(docs, embedding_function, collection_name=self.index_name,
                                                      path=index_dir)

            else:
                raise ValueError(
                    'Invalid environment value: Expecting one of memory, disk, on_premise or cloud')

        elif self.vector_store == 'array':
            vector_index = DocArrayInMemorySearch.from_documents(docs, embedding_function, index_name=persist_directory)
            return vector_index

        elif self.vector_store == 'faiss':
            vector_index = FAISS.from_documents(docs, embedding_function)
            return vector_index.save_local(persist_directory)

        else:
            raise ValueError(
                'Invalid vector_store value: Expecting one of chroma, milvus, qdrant, faiss or array')