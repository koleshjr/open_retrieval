import os
from typing import List, Optional
import pandas as pd
from src.open_retrieval.document_loaders import DocumentLoader
from src.open_retrieval.text_splitters import TextSplitter
from src.open_retrieval.embedding_providers import EmbeddingProvider
from src.open_retrieval.vector_databases import VectorDatabase
from src.open_retrieval.retrievers import Retriever
from rerankers import Reranker

def retrieval_pipeline(query: str, data_path: str,loader: str, splitter: str, vector_database:str, embedding_function: str, retrieval_type: str, index_name: str, filter_field: Optional[str]=None) -> List[str]:
    """
    Takes in a query and a list of documents paths and returns a list of top 5 documents that are related to the query
    """
    index_dir = 'tests/index/'

    if not os.path.exists(os.path.join(index_dir,index_name)):
        all_documents = []
        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            data = loader.load(filepath)
            print(filename.split('.')[0])
            extra_metadata = {"file_name": filename.split('.')[0]}
            documents = splitter.split(data, chunk_size = 800, chunk_overlap=0, extra_metadata=extra_metadata)
            all_documents.extend(documents)
        
        vector_index = vector_database.create_index(embedding_function=embedding_function,docs=all_documents, index_name=index_name,index_dir=index_dir)
    else:
        vector_index = vector_database.create_index(embedding_function=embedding_function, index_name=index_name,index_dir=index_dir)

    if filter_field:
        filter_params = {'file_name': filter_field}

    if retrieval_type == 'naive':
        retriever = Retriever(vector_index=vector_index)
        results = retriever.naive_retrieval(query = query, top_k=5, filter = filter_params )
    elif retrieval_type == 'ranked':
        retriever = Retriever(vector_index=vector_index, ranker = ranker)
        results = retriever.ranked_retrieval( query=query, top_k=15, ranked_top_k=5, filter = filter_params )
    
    return results

if __name__ == "__main__":
    csv_path = 'data/Test.csv'
    data_path = 'data/rag_data'
    retrieval_type = 'ranked'
    embedding_provider = 'huggingface' #fastembed
    database = 'chroma'
    ranking_model = "colbert"
    splitter_choice = 'recursive'
    index_name = f"{database}_index_{embedding_provider}"

    loader = DocumentLoader()
    splitter = TextSplitter(splitter=splitter_choice)
    ranker = Reranker(ranking_model, verbose=0)
    vector_database = VectorDatabase(vector_store=database)
    embedding_provider = EmbeddingProvider(embedding_provider=embedding_provider)
    embedding_function = embedding_provider.get_embedding_function()

    df = pd.read_csv(csv_path)   
    results = df.apply(lambda row: retrieval_pipeline(query=row['Query text'],data_path=data_path, loader=loader, splitter=splitter, vector_database=vector_database, embedding_function=embedding_function,index_name=index_name, retrieval_type=retrieval_type, filter_field=row['Document Title']), axis=1)
    results_df = pd.DataFrame({
        'Query No': df['Query No'],
        'Query text': df['Query text'],
        'Document No': df['Document No'],
        'Document Title': df['Document Title'],
        'Output_1': results.apply(lambda x: x[0]),
        'Output_2': results.apply(lambda x: x[1]),
        'Output_3': results.apply(lambda x: x[2]),
        'Output_4': results.apply(lambda x: x[3]),
        'Output_5': results.apply(lambda x: x[4])
    })
    os.makedirs('output',exist_ok=True)
    results_df.to_csv(f'output/results_{index_name}.csv', index=False)
    


