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
    
    def ranked_retrieval_with_rephrasing(self, query: str, rephrasing_model: str, ranking_model:str, top_k: int =5):
        """
        Retrieval With reranking and rephrasing
        """
        llm = Llm(model_provider='ollama', model_name=rephrasing_model).get_chat_model()
        rephrasor = Rephrasor(llm=llm, prompt=Config.rephrasing_prompt)
        pred = rephrasor.predict_json(query)
        docs= []
        for new_query in pred['result']['rephrased_query']:
            results = self.ranked_retrieval(query=new_query, ranking_model=ranking_model, top_k=top_k)
            docs.extend(results)

        ranker = Reranker(ranking_model, verbose = 0)
        data = ranker.rank(query = query, docs = [doc.page_content for doc in docs])
        results = next(item for item in data if item[0] == 'results')[1]
        sorted_results = sorted(results, key=lambda x: x.rank)
        top_5_texts = [result.text for result in sorted_results[:5]]
        return  top_5_texts
    
    def ranked_retrieval_with_llm(self, query: str, classifier_model:str, ranking_model:str, top_k: int =15):
        """
        Retrieval and using an llm in ranking
        """
        llm = Llm(model_provider='ollama', model_name=classifier_model).get_chat_model()
        classifier = Classifier(llm=llm, prompt=Config.classifier_prompt)
        docs =self.vector_database.similarity_search(query=query, k=top_k)
        yes_results =[]
        for doc in docs:
            pred = classifier.predict_json(question = query, content=doc.page_content)
            if pred['result']['label'] == 'yes':
                yes_results.append(doc)
        
        if len(yes_results)>5:
            ranker = Reranker(ranking_model, verbose=0)
            data = ranker.rank(query = query, docs = [doc.page_content for doc in yes_results])
            results = next(item for item in data if item[0] == 'results')[1]
            sorted_results = sorted(results, key=lambda x: x.rank)
            top_5_texts = [result.text for result in sorted_results[:5]]
        else:
            top_5_texts = [doc.page_content for doc in results]

        return top_5_texts






