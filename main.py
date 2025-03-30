import os
import sys
import streamlit as st
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data_loader.doc_indexer import retrieve_documents
from src.data_loader.unstructured_loader import DataIndexer
from src.ai_functions.prompt_functions import (
    generate_multiple_queries, 
    analyze_document_content,
    sentence_transformer_embeddings
)

# Load environment variables
load_dotenv()

# Setup logging
logger.add("logs/app.log", rotation="500 MB")

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Material Science Assistant",
        page_icon="🧪",
        layout="wide"
    )
    
    # Page title and description
    st.title("Material Science Document Assistant")
    st.write("Upload material science documents and query their content with AI-powered analysis")
    
    # Initialize session state for storing document indices
    if 'document_indices' not in st.session_state:
        st.session_state.document_indices = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Documents")
        st.write("Supported formats: PDF, DOCX, DOC, TXT")
        uploaded_files = st.file_uploader("Choose files to analyze", 
                                         accept_multiple_files=True,
                                         type=["pdf", "docx", "doc", "txt"])
        
        if uploaded_files:
            process_button = st.button("Process Documents")
            if process_button:
                with st.spinner("Processing documents... This may take a few minutes depending on file size."):
                    process_uploaded_documents(uploaded_files)
                    st.sidebar.success(f"✅ Successfully processed {len(uploaded_files)} document(s)")
    
    # Query interface in main area
    st.header("Ask About Your Materials Science Documents")
    
    # Only show query interface if documents have been processed
    if st.session_state.document_indices:
        query = st.text_input("Enter your materials science question:")
        
        if query:
            with st.spinner("Analyzing materials science documents..."):
                # Generate multiple related queries behind the scenes
                related_queries = generate_multiple_queries(query)
                
                # Retrieve relevant content for each query without showing intermediate steps
                all_results = []
                
                # Process each document index
                for index_path in st.session_state.document_indices:
                    try:
                        document_name = Path(index_path).name
                        logger.info(f"Searching in document: {document_name}")
                        
                        # Retrieve documents for each query (original query and generated queries)
                        for q in [query] + related_queries:
                            doc_results = retrieve_documents(
                                embeddings=sentence_transformer_embeddings,
                                query=q,
                                document_name=document_name,
                                search_type="mmr",
                                k=5
                            )
                            all_results.extend(doc_results)
                    except Exception as e:
                        logger.error(f"Error retrieving documents: {str(e)}")
                
                analysis = analyze_document_content(query, all_results)
                
                # Display the final analysis
                st.markdown(analysis)
    else:
        st.info("Please upload and process documents to begin your materials science exploration.")

def process_uploaded_documents(uploaded_files):
    """Process uploaded documents and create vector indices"""
    try:
        # Create a temporary directory to store uploaded files
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded files to the temporary directory
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))
        
        # Create DataIndexer and process files
        indexer = DataIndexer(
            embeddings=sentence_transformer_embeddings,
            input_paths=file_paths,
            max_workers=4
        )
        
        index_paths = indexer.process_all_files()
        
        # Update session state with new indices
        st.session_state.document_indices.extend(index_paths)
        
        # Display summary
        summary = indexer.get_summary()
        
        if summary['failed']['count'] > 0:
            st.sidebar.warning(f"⚠️ Failed to process {summary['failed']['count']} documents.")
            for failed_file in summary['failed']['files']:
                st.sidebar.write(f"- {Path(failed_file).name}")
            
    except Exception as e:
        logger.error(f"Error processing uploaded documents: {str(e)}")
        st.sidebar.error(f"Error processing documents: {str(e)}")

if __name__ == "__main__":
    main()