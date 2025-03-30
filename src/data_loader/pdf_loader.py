import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from loguru import logger

def load_pdf_with_pdfloader(document_path: str):
    """Load a PDF using PyPDFLoader from langchain_community."""
    try:
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        logger.info(f"Successfully loaded PDF with PyPDFLoader: {document_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF with PyPDFLoader: {document_path}, Error: {str(e)}")
        return None

def load_pdf_with_pdfplumber(document_path: str):
    """Load a PDF using pdfplumber."""
    try:
        documents = []
        with pdfplumber.open(document_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": document_path, "type": "pdf"}))
        logger.info(f"Successfully loaded PDF with pdfplumber: {document_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF with pdfplumber: {document_path}, Error: {str(e)}")
        return None

def load_pdf(document_path: str):
    """
    Attempts to load a PDF file using both PyPDFLoader and pdfplumber, 
    returning the first successful result.
    
    Args:
        document_path (str): The path to the PDF document.
    
    Returns:
        List[Document]: A list of Document objects if successful, or None if both methods fail.
    """
    # First try PyPDFLoader
    documents = load_pdf_with_pdfloader(document_path)
    if documents:
        return documents
    
    # If PyPDFLoader fails, try pdfplumber
    documents = load_pdf_with_pdfplumber(document_path)
    if documents:
        return documents
    
    # If both fail, return None
    return None


if __name__ == "__main__":
    document_path = "sample_data/Fabrice Grinda, Founding Partner at FJ Labs — Serial Entrepreneur & Investor in 700 Startups! _ by Miguel Armaza _ Wharton FinTech _ Medium.pdf"  # Example document path

    documents = load_pdf(document_path)
    if documents:
        logger.info(f"Successfully loaded {len(documents)} documents from {document_path}")
    else:
        logger.error(f"Failed to load PDF document from {document_path}")
