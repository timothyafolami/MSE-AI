#!/usr/bin/env python3

import os
import sys
from loguru import logger

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Setup logging
logger.add("logs/retrieval_test.log", rotation="500 MB")

# Import project modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.data_loader.doc_indexer import retrieve_documents
from src.data_loader import settings

def test_document_retrieval():
    """
    Test document retrieval from the materials database.
    """
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Test queries
    test_queries = [
        "What materials have high corrosion resistance in marine environments?",
        "Which metals have the highest strength to weight ratio?",
        "Materials suitable for high temperature applications above 500°C",
        "What are the properties of titanium alloys?",
        "Materials with good weldability and machinability"
    ]
    
    print("\n" + "="*80)
    print("MATERIAL DATABASE RETRIEVAL TEST")
    print("="*80 + "\n")
    
    # Ensure the index exists
    unified_index_path = os.path.join(settings.DOC_INDEXES_DIR, "materials_database")
    if not os.path.exists(os.path.join(unified_index_path, "index.faiss")):
        logger.error(f"Unified materials database index not found at {unified_index_path}")
        print(f"ERROR: Unified database not found at {unified_index_path}")
        print("Please run: python index_data.py --rebuild")
        return
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)
        
        try:
            # Retrieve documents
            docs = retrieve_documents(
                embeddings=embeddings,
                query=query,
                search_type="mmr",
                k=2  # Limit to 2 results for clarity
            )
            
            # Display results
            if docs:
                print(f"Retrieved {len(docs)} document segments:")
                for j, doc in enumerate(docs, 1):
                    # Extract metadata
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    source = metadata.get('source', 'Unknown source')
                    doc_name = metadata.get('doc_name', 'Unknown document')
                    
                    # Print document content (limited to 500 chars for readability)
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    print(f"\nResult {j} [Source: {doc_name}]:")
                    print(f"{content[:500]}...")
                    if len(content) > 500:
                        print("(content truncated for readability)")
            else:
                print("No results found.")
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {str(e)}")
            print(f"ERROR: {str(e)}")
    
    print("\n" + "="*80)
    print("Test completed.")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_document_retrieval()