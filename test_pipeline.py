import os
from typing import List
from src.openrag.document_loaders import DocumentLoader
from src.openrag.text_splitters import TextSplitter
from src.openrag.embedding_providers import EmbeddingProvider
from src.openrag.vector_databases import VectorDatabase
from src.openrag.retrievers import Retriever
def retrieval_pipeline(query: str,folder_path: str,  documents: List[str]) -> List[str]:
    """
    Takes in a query and a list of documents paths and returns a list of top 5 documents that are related to the query
    """
    loader = DocumentLoader()
    splitter = TextSplitter(splitter='recursive')
    all_documents = []
    for filename in documents:
        filepath = os.path.join(folder_path, filename)
        data = loader.load(filepath)
        documents = splitter.split_to_documents(data, chunk_size = 768, chunk_overlap=120)
        all_documents.extend(documents)

    embedding_provider = EmbeddingProvider(embedding_provider='huggingface')
    embedding_function = embedding_provider.get_embedding_function()
    index_name = "test_faiss"
    index_dir = 'tests/index/'

    vector_database = VectorDatabase(vector_store='faiss')
    vector_index = vector_database.create_index(embedding_function=embedding_function,docs=all_documents, index_name=index_name,index_dir=index_dir)

    retriever = Retriever(vector_index=vector_index)
    # results = retriever.naive_retrieval(query = query, top_k=5 )
    results = retriever.ranked_retrieval( query=query,ranking_model="colbert" )
    return results

if __name__ == "__main__":
    query = "explain self attention?"
    folder_path = "tests/test_data"
    documents = ["attention.pdf"]
    results = retrieval_pipeline(query=query , folder_path=folder_path, documents=documents)
    print(results)
    # for x in results:
    #     print(x.page_content)
    #     print("*"*100)