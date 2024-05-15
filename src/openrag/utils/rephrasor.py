from operator import itemgetter
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class RephrasedQuery(BaseModel):
    """a list of rephrased queries string"""
    rephrased_queries: List[str] = Field(...,description="a list of 4 rephrased queries to be extracted")

class Rephrasor:
    def __init__(self, llm, prompt):
        self.model = llm
        self.parser = JsonOutputParser(pydantic_object = RephrasedQuery)
        self.custom_prompt = PromptTemplate(template = prompt, 
                                           input_variables = ["query"],   
                                           partial_variables = {"format_instructions": self.parser.get_format_instructions()},                              
        )

    def predict_json(self, page_text):

        chain = (
            {"query": itemgetter("query")}
            | self.custom_prompt
            | self.model 
            | self.parser
        )
        chain = chain.with_retry()
        return chain.invoke({"query": page_text})
    