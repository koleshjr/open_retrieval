from typing import Optional 
from langchain_community.chat_models import ChatOllama
class Llm:
    def __init__(self, model_provider: str, model_name: Optional[str] = None):
        self.model_provider = model_provider 
        self.model_name = model_name 
    def get_chat_model(self):
      if self.model_provider == 'ollama':
        return ChatOllama(model = self.model_name)
      else:
        raise Exception("Invalid model provider we currently support only ollama models")