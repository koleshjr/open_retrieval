import os
from typing import List
import pandas as pd
from pathlib import Path
from src.openrag.document_loaders import DocumentLoader
from src.openrag.text_splitters import TextSplitter
from src.openrag.embedding_providers import EmbeddingProvider
from src.openrag.vector_databases import VectorDatabase
from src.openrag.retrievers import Retriever
from typing import Optional

def retrieval_pipeline(query: str, data_path: str,  retrieval_type: str, embedding_provider: str, database: str, filter_field: Optional[str]=None) -> List[str]:
    """
    Takes in a query and a list of documents paths and returns a list of top 5 documents that are related to the query
    """
    index_name = f"{database}_index_{embedding_provider}"
    index_dir = 'tests/index/'
    vector_database = VectorDatabase(vector_store=database)
    embedding_provider = EmbeddingProvider(embedding_provider=embedding_provider)
    embedding_function = embedding_provider.get_embedding_function()

    if not os.path.exists(os.path.join(index_dir,index_name)):
        loader = DocumentLoader()
        splitter = TextSplitter(splitter='recursive')
        all_documents = []
        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            data = loader.load(filepath)
            extra_metadata = {"file_name": filename}
            documents = splitter.split_to_documents(data, chunk_size = 800, chunk_overlap=0, extra_metadata=extra_metadata)
            all_documents.extend(documents)
        
        vector_index = vector_database.create_index(embedding_function=embedding_function,docs=all_documents, index_name=index_name,index_dir=index_dir)
    else:
        vector_index = vector_database.create_index(index_name=index_name,index_dir=index_dir)

    if filter_field:
        filter_params = {'file_name': filter_field}

    retriever = Retriever(vector_index=vector_index)
    if retrieval_type == 'naive':
        results = retriever.naive_retrieval(query = query, top_k=5, filter = filter_params )
    elif retrieval_type == 'ranked':
        results = retriever.ranked_retrieval( query=query,ranking_model="colbert", filter = filter_params )


    return results

if __name__ == "__main__":
    csv_path = 'data/Test.csv'
    data_path = 'data/rag_data'
    retrieval_type = 'ranked'
    embedding_provider = 'qdrant'
    database = 'chroma'
    df = pd.read_csv(csv_path)   
    results = df.apply(lambda row: retrieval_pipeline(query=row['Query text'], data_path=data_path, retrieval_type=retrieval_type, embedding_provider=embedding_provider, database=database, filter_field=row['Document Title']), axis=1)
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
    results_df.to_csv(f'output/results_{embedding_provider}_{database}.csv', index=False)
    


