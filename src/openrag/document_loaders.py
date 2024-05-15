import bs4
import logging
from typing import Optional
from langchain_community.document_loaders.csv_loader import CSVLoader   
from langchain_community.document_loaders import JSONLoader, PyPDFLoader, WebBaseLoader,UnstructuredWordDocumentLoader,UnstructuredMarkdownLoader,UnstructuredHTMLLoader, UnstructuredPowerPointLoader

logging.basicConfig(level=logging.INFO)
class DocumentLoader:
    """
    This class is used to load documents from different sources and return them as a list of strings.
    The supported sources include CSV, JSON, PDF, HTML, Markdown, Word and Powerpoint documents.
    """
    def __init__(self) -> None:
        """
        Initialize the DocumentLoader class.
        """
        self.logger = logging.getLogger(__name__)
    
    def load_and_get_data(self, file_path: Optional[str]= None, url_path: Optional[str] = None):
            
        """
        Load a document from a file or URL and return its data.

        Args:
            file_path (Optional[str]): The path to the file.
            url_path (Optional[str]): The URL of the document.

        Returns:
            Any: The data from the document.

        Raises:
            ValueError: If neither file_path nor url_path is provided.
        """
        if file_path is not None:
            try:
                if file_path.endswith('.csv'):
                    loader = CSVLoader(file_path)
                elif file_path.endswith('.json'):
                    loader = JSONLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith('.html') or file_path.endswith('.htm'):
                    loader = UnstructuredHTMLLoader(file_path)
                elif file_path.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(file_path)
                elif file_path.endswith('.docx') or file_path.endswith('.doc'):
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif file_path.endswith('pptx'):
                    loader = UnstructuredPowerPointLoader(file_path)

                data = loader.load_and_split()
                return data
            except Exception as e:
                self.logger.error(f"Error loading the file: {e}")
                return "Error loading the file. We currently support csv files, json files, pdf files , powerpoint and html files"
            
        elif url_path is not None:
            try:
                bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
                loader = WebBaseLoader(web_paths = (url_path), bs_kwargs = {"parse_only": bs_strainer})
                data = loader.load()
                return data
            except Exception as e:
                self.logger.error(f"Error loading the url: {e}")
                return "Error loading the url. Please make sure you have provided a valid url"
            