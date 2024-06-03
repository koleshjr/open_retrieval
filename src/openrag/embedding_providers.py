from typing import Optional, Union
from langchain_community.embeddings import FastEmbedEmbeddings, HuggingFaceEmbeddings, OllamaEmbeddings

class EmbeddingProvider:
    """
    A class that provides different embedding functions based on the embedding_provider specified in the settings file.
    """
    def __init__(self, embedding_provider: str):
        """
        Initialize the EmbeddingProvider class.

        Args:
            embedding_provider (str): The name of the embedding provider.
        """
        self.embedding_provider = embedding_provider

    def get_embedding_function(self, model_name: Optional[str] = None)-> Union[FastEmbedEmbeddings, HuggingFaceEmbeddings, OllamaEmbeddings]:
        """
        Get the embedding function based on the embedding_provider and model_name.

        Args:
            model_name (str, optional): The name of the model. Defaults to None.

        Returns:
            The embedding function.
        """
        if self.embedding_provider == "huggingface":
            if model_name:
                return HuggingFaceEmbeddings(model_name=model_name)
            else:
                model_kwargs = {'trust_remote_code': True}
                return HuggingFaceEmbeddings(model_name='Alibaba-NLP/gte-large-en-v1.5',model_kwargs = model_kwargs)

        elif self.embedding_provider == "ollama":
            if model_name:
                return OllamaEmbeddings(model_name)
            else:
                return OllamaEmbeddings()


        elif self.embedding_provider == 'fastembed':
            # Assuming FastEmbedEmbeddings doesn't require any API key
            if model_name:
                return FastEmbedEmbeddings(model_name = model_name)
            else:
                return FastEmbedEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

        else:
            raise ValueError(
                f"Embedding provider {self.embedding_provider} is not supported. We currently support huggingface, fastembed and ollama as embedding providers")