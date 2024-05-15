from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class ClassifierOutput(BaseModel):
    """a list of rephrased queries string"""
    related: bool = Field(...,description="can the content be used to answer the question or not?")

class Classifier:
    def __init__(self, llm, prompt):
        self.model = llm
        self.parser = JsonOutputParser(pydantic_object = ClassifierOutput)
        self.custom_prompt = PromptTemplate(template = prompt, 
                                           input_variables = ["question", "content"],   
                                           partial_variables = {"format_instructions": self.parser.get_format_instructions()},                              
        )

    def predict_json(self, question, content):

        chain = (
            {"query": itemgetter("question")}
            | {"content": itemgetter("content")}
            | self.custom_prompt
            | self.model 
            | self.parser
        )
        chain = chain.with_retry()
        return chain.invoke({"question": question, "content": content})
    