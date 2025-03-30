import os
import sys

import docx
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.data_loader import settings
from loguru import logger
from src.data_loader.pdf_loader import load_pdf 

def create_and_save_document_index(embeddings, document_path):
    """
    Processes documents (PDF, DOCX/DOC, TXT), creates embeddings, and saves the FAISS index.
    
    Args:
        embeddings: The embeddings object to use
        document_path (str): Path to the document
        
    Returns:
        str: Path where the index was saved
    """
    # Ensure output directory exists
    os.makedirs(settings.DOC_INDEXES_DIR, exist_ok=True)

    # Get base document name for saving
    doc_name = os.path.splitext(os.path.basename(document_path))[0]
    save_path = os.path.join(settings.DOC_INDEXES_DIR, doc_name)

    # Load document based on file type
    file_ext = os.path.splitext(document_path)[1].lower()
    try:
        if file_ext == '.pdf':
            # Use the load_pdf function from pdf_loader.py
            documents = load_pdf(document_path)
        elif file_ext in ['.docx', '.doc']:
            doc = docx.Document(document_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            documents = [Document(page_content=text, metadata={"source": document_path, "type": "word"})]
        elif file_ext == '.txt':
            loader = TextLoader(document_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading document {document_path}: {str(e)}")
        raise ValueError(f"Failed to load or process document: {document_path} due to {str(e)}")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,  # Increased overlap for better context preservation
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_documents = text_splitter.split_documents(documents)
    
    logger.info(f"Split document into {len(split_documents)} chunks")

    try:
        # Create vector store from split documents
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        
        # Ensure directory exists for saving the index
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the FAISS index
        vectorstore.save_local(save_path)
    except Exception as e:
        logger.error(f"Error creating or saving FAISS index for {document_path}: {str(e)}")
        raise ValueError(f"Failed to create or save index for document: {document_path} due to {str(e)}")
    
    logger.success(f"Successfully indexed {document_path} → {save_path}")
    return save_path

def retrieve_documents(
    embeddings,
    query: str,
    document_name: str,
    search_type: str = "mmr",
    k: int = 5
) -> list:
    """
    Retrieve relevant documents from a saved index based on a query.
    
    Args:
        embeddings: The embeddings object to use
        query (str): Search query or question
        document_name (str): Name of the document index to search
        search_type (str): Type of search ('mmr' or 'similarity')
        k (int): Number of documents to return
    
    Returns:
        List[str]: Relevant document chunks
    """
    # Load the saved index
    index_path = os.path.join(settings.DOC_INDEXES_DIR, document_name)
    try:
        logger.info(f"Loading index from {index_path}")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Failed to load index for {document_name}: {str(e)}")
        raise ValueError(f"Failed to load index for {document_name}: {str(e)}")

    # Create retriever with specified search parameters
    search_kwargs = {
        "k": k,
    }
    
    # Add fetch_k parameter for MMR search
    if search_type == "mmr":
        search_kwargs["fetch_k"] = min(k * 3, 100)  # Fetch more candidates for diversity
        search_kwargs["lambda_mult"] = 0.7  # Balance between relevance and diversity
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    # Perform document retrieval
    try:
        logger.info(f"Retrieving documents for query: {query}")
        docs = retriever.invoke(query)
        logger.success(f"Retrieved {len(docs)} documents for query: {query}")
    except Exception as e:
        logger.error(f"Error during document retrieval for {document_name}: {str(e)}")
        raise ValueError(f"Failed to retrieve documents for query '{query}' from {document_name}: {str(e)}")

    # Extract the content from retrieved documents
    output = [doc for doc in docs]
    return output