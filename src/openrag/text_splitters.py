import logging
from typing import Optional, List
from langchain.docstore.document import Document
from langchain.text_splitter import HTMLHeaderTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter,NLTKTextSplitter

class TextSplitter:
    """
    A class for splitting text into smaller pieces.

    Args:
        splitter (str): The name of the splitter to use.
        splitter_args (Optional[list]): The arguments to pass to the splitter.

    Attributes:
        splitter (str): The name of the splitter to use.
        splitter_args (Optional[list]): The arguments to pass to the splitter.
        logger (logging.Logger): A logger for logging messages.
    """
    def __init__(self, splitter: str, splitter_args: Optional[list] = None):
        self.splitter = splitter
        self.splitter_args = splitter_args
        self.logger = logging.getLogger(__name__)

    def create_documents(self, texts):
        """
        Creates a list of Document objects from a list of text strings.

        Args:
            texts (List[str]): A list of text strings.

        Returns:
            List[Document]: A list of Document objects.
        """
        docs = []
        for text in texts:
            doc = Document(
                page_content = text
            )
            docs.append(doc)
        return docs

    def modify_splitter(self, data, splitter):
        """
        Modifies the splitter based on the input data and the splitter type.

        Args:
            data (list): A list of documents.
            splitter (object): A text splitter object.

        Returns:
            list: A list of modified documents.
        """
        results = []
        for doc in data:
            texts = splitter.split_text(doc.page_content)
            if self.splitter in ['htmlheader', 'markdownheader']:
                documents = texts
            else:
                documents = splitter.create_documents(texts)

            for document in documents:
                document.metadata = doc.metadata

            results.extend(documents)
        return results

    def split_to_documents(self, data: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200, embedding_function: Optional[str] = None):
        """
        Splits a list of Documents into smaller Documents based on the selected splitter.

        Args:
            data (List[Document]): A list of Documents to split.
            chunk_size (int, optional): The size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.
            embedding_function (Optional[str], optional): The name of the embedding function to use. Defaults to None.

        Returns:
            List[Document]: A list of split Documents.
        """
        if self.splitter == "htmlheader":
            #https://python.langchain.com/docs/modules/data_connection/document_transformers/HTML_header_metadata
            splitter = HTMLHeaderTextSplitter(headers_to_split_on= self.splitter_args)
            return self.modify_splitter(data=data, splitter=splitter)
        
        elif self.splitter == "character":
            #https://python.langchain.com/docs/modules/text_splitter/character_text_splitter
            splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            return self.modify_splitter(data=data, splitter=splitter)
        
        elif self.splitter == "markdownheader":
            #https://python.langchain.com/docs/modules/data_connection/document_transformers/markdown_header_metadata
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on= self.splitter_args)
            return self.modify_splitter(data=data, splitter=splitter)
            
        elif self.splitter == "recursive":
            #https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
            splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            return self.modify_splitter(data=data, splitter=splitter)

        elif self.splitter == "token":
            #https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token
            splitter = NLTKTextSplitter(chunk_size = chunk_size)
            return self.modify_splitter(data=data, splitter=splitter)
        
        else:
            raise ValueError('Invalid splitter value: Expecting one of htmlheader, character, markdown, recursive, semantic, or nltk')