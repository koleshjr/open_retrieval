from rerankers import Reranker
from src.openrag.utils.llms import Llm
from src.openrag.utils.rephrasor import Rephrasor
from src.openrag.utils.classifier import Classifier
from src.openrag.utils.config import Config
class Retriever:
    def __init__(self, vector_index):
        self.vector_database = vector_index
        
    def naive_retrieval(self, query: str, top_k: int = 5 ):
        """
        Naive Retrieval
        """
        top_k_results = self.vector_database.similarity_search(query=query, k=top_k)
        return top_k_results
    
    def ranked_retrieval(self, query: str,ranking_model:str, top_k: int = 15, ):
        """
        Retrieval With reranking
        """
        ranker = Reranker(ranking_model, verbose=0)
        docs =self.vector_database.similarity_search(query=query, k=top_k)
        data = ranker.rank(query = query, docs = [doc.page_content for doc in docs])
        # Extract the list of Result objects
        results = next(item for item in data if item[0] == 'results')[1]

        # Sort results by rank
        sorted_results = sorted(results, key=lambda x: x.rank)

        # Extract the text fields from the top 5 Result objects
        top_5_texts = [result.text for result in sorted_results[:5]]
        return  top_5_texts
    







