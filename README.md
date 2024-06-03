# **openrag**
![OPENRAG](images\openrag.jpg)
* A Python package based on langchain that abstracts RAG (Retrieval-Augmented Generation) utilities, providing unified document loaders, embedding models, text splitters, vector databases and retrievers in one package based on open source models.
* No api keys needed

## **BENEFITS**
* Unified Interface: Simplifies complex processes by providing a unified interface for various RAG utilities.
* Flexibility: Supports multiple document sources, splitting methods, embedding providers, vector databases, retrievers, enhancing adaptability to diverse use cases.
* Scalability: Designed to accommodate future functionalities, ensuring long-term viability and relevance.

## **INSTALL AND RUN**
    pip install openrag

### **Document Loaders**
The DocumentLoaders class is used to load documents from different sources and return them as a list of strings. The supported sources include csv files, json files, pdf files , html_files, markdown, word_documents, and powerpoint files.

The purpose of the DocumentLoader class is to provide a single interface for loading documents from various sources and to ensure that the process of loading documents is consistent across different sources. This allows the code that uses the DocumentLoader class to be more flexible and easier to maintain.

By using the DocumentLoader class, the code can be written in a way that is independent of the specific source of the document. This makes it easier to modify or extend the code in the future, as new sources of documents can be added without affecting the rest of the code.

#### Example usage:
    from openrag.document_loaders import DocumentLoader
    loader = DocumentLoader()
    file_path = os.path.join("path/to/file.csv")
    data = loader.load(file_path)
    url = "https://example.com/document.html"
    data = loader.load(url)

### **Text Splitters**
The TextSplitters class is used to split text into a list of document objects. It can be used to preprocess the text data before indexing it to a vector database.

The TextSplitter class provides a number of different splitters, including splitters that split based on  htmlheader, characters, markdownheader, paragraphs(recursive), or tokens. The splitters can be configured with arguments to control the splitting process, such as the maximum length of the chunks or the set of headers to split on and also include extra metadata in the resulting chunks.

The TextSplitter class is designed to be flexible and can be used with a wide range of text data, including HTML documents, Markdown documents, and plain text. It is also designed to be scalable in future.

#### Example usage:
    from openrag.document_loaders import DocumentLoader
    from openrag.text_splitters import TextSplitter
    document_loader = DocumentLoader()
    splitter = TextSplitter(splitter="recursive")

    file_path = os.path.join("path/to/file.csv")
    data = loader.load(file_path)
    documents = splitter.split(data, chunk_size = 800, chunk_overlap=0, extra_metadata=extra_metadata)

### **Embedding Providers**
The EmbeddingProviders class is responsible for providing different embedding functions based on the embedding_provider specified. It initializes the class with the specified embedding_provider and provides the get_embedding_function method to retrieve the embedding function based on the model_name.

At the moment we currently support embedding models from huggingface, fastembed and ollama.

#### Example usage:
    from openrag.document_loaders import DocumentLoader
    from openrag.text_splitters import TextSplitter
    from openrag.embedding_providers import EmbeddingProvider

    loader = DocumentLoader()
    file_path = os.path.join("path/to/file.csv")
    data = loader.load(file_path)
    url = "https://example.com/document.html"
    data = loader.load(url)

    # Split the text
    splitter = TextSplitter(splitter="recursive")
    documents = splitter.split(data, chunk_size = 800, chunk_overlap=0, extra_metadata=extra_metadata)

    # Get the embedding function
    embedding_type = 'huggingface' #fastembed
    embedding_provider = EmbeddingProvider(embedding_provider=embedding_type)
    embedding_function = embedding_provider.get_embedding_function()

#### **Vector Databases**
The purpose of the VectorDatabase class is to manage different vector databases, such aschroma, milvus, qdrant, faiss or array. It provides a consistent interface for creating and managing indexes for different vector databases.

#### Example usage
    from openrag.document_loaders import DocumentLoader
    from openrag.text_splitters import TextSplitter
    from openrag.embedding_providers import EmbeddingProvider
    from openrag.vector_databases import VectorDatabase

    loader = DocumentLoader()
    file_path = os.path.join("path/to/file.csv")
    data = loader.load(file_path)
    url = "https://example.com/document.html"
    data = loader.load(url)

    # Split the text
    splitter = TextSplitter(splitter="recursive")
    all_documents = splitter.split(data, chunk_size = 800, chunk_overlap=0, extra_metadata=extra_metadata)

    # Get the embedding function
    embedding_type = 'huggingface' #fastembed
    embedding_provider = EmbeddingProvider(embedding_provider=embedding_type)
    embedding_function = embedding_provider.get_embedding_function()
    
    # embed the documents
    database = 'chroma' #type of the database
    index_dir = 'tests/index/' #path to the folder where the indexes are stored
    index_name = f"{database}_index_{embedding_provider}" #name of the index
    vector_database = VectorDatabase(vector_store=database)
    vector_database.create_index(embedding_function, documents, index_dir)
    vector_index = vector_database.create_index(embedding_function=embedding_function,docs=all_documents, index_name=index_name,index_dir=index_dir)

### **Retrievers**
The purpose of the Retriever class is to manage different retrival techniques such as naive_retrieval and ranked_retrieval. It provides a consistent interface for creating and managing different retrival techniques
It uses the unified rerankers API by answerdotai : https://github.com/AnswerDotAI/rerankers

#### Example usage
    from openrag.document_loaders import DocumentLoader
    from openrag.text_splitters import TextSplitter
    from openrag.embedding_providers import EmbeddingProvider
    from openrag.vector_databases import VectorDatabase
    from openrag.retrievers import Retriever
    from rerankers import Reranker

    loader = DocumentLoader()
    file_path = os.path.join("path/to/file.csv")
    data = loader.load(file_path)
    url = "https://example.com/document.html"
    data = loader.load(url)

    # Split the text
    splitter = TextSplitter(splitter="recursive")
    all_documents = splitter.split(data, chunk_size = 800, chunk_overlap=0, extra_metadata=extra_metadata)

    # Get the embedding function
    embedding_type = 'huggingface' #fastembed
    embedding_provider = EmbeddingProvider(embedding_provider=embedding_type)
    embedding_function = embedding_provider.get_embedding_function()
    
    # embed the documents
    database = 'chroma' #type of the database
    index_dir = 'tests/index/' #path to the folder where the indexes are stored
    index_name = f"{database}_index_{embedding_provider}" #name of the index
    vector_database = VectorDatabase(vector_store=database)
    vector_database.create_index(embedding_function, documents, index_dir)
    vector_index = vector_database.create_index(embedding_function=embedding_function,docs=all_documents, index_name=index_name,index_dir=index_dir)

    # retrieve top-k contexts
    retrieval_type = 'ranked' #type of retrieval: naive or ranked
    ranking_model = "colbert" #ranking model to choose from
    filter_params = {'file_name': 'who_guidelines'} #filter parameters 
    ranker = Reranker(ranking_model, verbose=0)
    retriever = Retriever(vector_index=vector_index, ranker = ranker)
    results = retriever.ranked_retrieval( query=query, top_k=15, filter = filter_params )

## **CONTRIBUTE**
Feel free to contribute to openrag by submitting bug reports, feature requests, or pull requests on GitHub.

