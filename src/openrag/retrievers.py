from rerankers import Reranker
from typing import Optional, Dict
class Retriever:
    def __init__(self, vector_index, ranker: Optional[Reranker] = None):
        self.vector_database = vector_index
        self.reranker = ranker
        
    def naive_retrieval(self, query: str, top_k: int = 5, filter: Optional[Dict[str, str]] = None ):
        """
        Naive Retrieval
        """
        top_k_results = self.vector_database.similarity_search(query=query, k=top_k, filter=filter)
        return [doc.page_content for doc  in top_k_results]
    
    def ranked_retrieval(self, query: str, top_k: int = 15, ranked_top_k: int = 5, filter: Optional[Dict[str, str]] = None):
        """
        Retrieval With reranking
        """
        docs =self.vector_database.similarity_search(query=query, k=top_k, filter = filter)
        data = self.reranker.rank(query = query, docs = [doc.page_content for doc in docs])
      
        # Sort results by rank
        sorted_results = sorted(data.results, key=lambda x: x.rank)

        # Extract the text fields from the top 5 Result objects
        top_5_texts = [result.text for result in sorted_results[:ranked_top_k]]
        return  top_5_texts

