import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.openrag.document_loaders import DocumentLoader

class TestDocumentLoader:
    @pytest.fixture
    def document_loader(self):
        return DocumentLoader()
    
    def test_load_and_get_data_from_csv(self, document_loader):
        file_path = "tests\test_data\FuelConsumption.csv"
        data = document_loader.load_and_get_data(file_path=file_path)
        assert data is not None

    def test_load_and_get_data_from_pdf(self, document_loader):
        file_path = "tests\test_data\attention.pdf"
        data = document_loader.load_and_get_data(file_path=file_path)
        assert data is not None

    def test_load_and_get_data_from_html(self, document_loader):
        file_path = "tests\test_data\html5.html"
        data = document_loader.load_and_get_data(file_path=file_path)
        assert data is not None

    def test_load_and_get_data_from_docx(self, document_loader):
        file_path = "tests\test_data\attention.docx"
        data = document_loader.load_and_get_data(file_path=file_path)
        assert data is not None

    def test_load_and_get_data_from_json(self, document_loader):
        file_path = "tests\test_data\latestblock.json"
        data = document_loader.load_and_get_data(file_path=file_path)
        assert data is not None

    def test_load_and_get_data_from_md(self, document_loader):
        file_path = "tests\test_data\bitcoin.md"
        data = document_loader.load_and_get_data(file_path=file_path)
        assert data is not None

    def test_load_and_get_data_from_url(self, document_loader):
        url_path = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        data = document_loader.load_and_get_data(url_path =url_path)
        assert data is not None

    def test_load_and_get_data_from_powerpoint(self, document_loader):
        url_path = "tests\test_data\intro_to_data_science.pptx"
        data = document_loader.load_and_get_data(url_path =url_path)
        assert data is not None

    