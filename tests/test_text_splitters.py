import os
import sys 
import pytest 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.openrag.document_loaders import DocumentLoader
from src.openrag.text_splitters import TextSplitter 

class TestTextSplitter:
    document_loader = DocumentLoader()

    # @pytest.fixture
    # def html_text_splitter(self):
    #     return TextSplitter(splitter='htmlheader', splitter_args=[("h1", "Header 1"),("h2", "Header 2"),("h3", "Header 3"),("h4", "Header 4"),])
    
    # def test_split_html(self,html_text_splitter):
    #     file_path = "tests/test_data/html5.html"
    #     data = self.document_loader.load(file_path=file_path)
    #     documents = html_text_splitter.split(data)
    #     assert len(documents) > 0

    @pytest.fixture
    def character_text_splitter(self):
        return TextSplitter(splitter='character')
    
    def test_split_character(self, character_text_splitter):
        file_path = "data/rag_data/1_WHO_FCTC.pdf"
        data = self.document_loader.load(file_path=file_path)
        documents = character_text_splitter.split(data)
        assert len(documents) > 0

    @pytest.fixture
    def markdown_text_splitter(self):
        return TextSplitter(splitter='markdownheader', splitter_args=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3")])
    
    def test_split_markdown(self, markdown_text_splitter):
        file_path = "tests/test_data/bitcoin.md"
        data = self.document_loader.load(file_path=file_path)
        documents = markdown_text_splitter.split(data)
        assert len(documents) > 0

    @pytest.fixture
    def recursive_text_splitter(self):
        return TextSplitter(splitter='recursive')
    
    def test_split_recursive(self, recursive_text_splitter):
        file_path = "data/rag_data/1_WHO_FCTC.pdf"
        data = self.document_loader.load(file_path=file_path)
        documents = recursive_text_splitter.split(data)
        assert len(documents) > 0

    @pytest.fixture
    def token_text_splitter(self):
        return TextSplitter(splitter='token')
    
    def test_split_token(self, token_text_splitter):
        file_path = "data/rag_data/4_GovStack_Specs.docx"
        data = self.document_loader.load(file_path=file_path)
        documents = token_text_splitter.split(data)
        assert len(documents) > 0