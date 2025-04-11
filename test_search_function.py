#!/usr/bin/env python3

import os
import sys
from loguru import logger

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Setup logging
logger.add("logs/search_test.log", rotation="500 MB")

# Import project modules
from src.data_loader import settings
from src.ai_functions.prompt_functions import search_materials_database

def test_search_materials_database():
    """
    Test the search_materials_database function directly.
    This tests the full retrieval pipeline that is used by the recommendation system.
    """
    # Test queries
    test_queries = [
        "What materials have high corrosion resistance in marine environments?",
        "Which metals have the highest strength to weight ratio?",
        "Materials suitable for high temperature applications above 500°C",
        "What are the properties of titanium alloys?",
        "Materials with good weldability and machinability"
    ]
    
    print("\n" + "="*80)
    print("MATERIALS DATABASE SEARCH TEST")
    print("="*80 + "\n")
    
    # Ensure the index exists
    unified_index_path = os.path.join(settings.DOC_INDEXES_DIR, "materials_database")
    if not os.path.exists(os.path.join(unified_index_path, "index.faiss")):
        logger.error(f"Unified materials database index not found at {unified_index_path}")
        print(f"ERROR: Unified database not found at {unified_index_path}")
        print("Please run: python index_data.py --rebuild")
        return
    
    # Test the search function
    try:
        # Search using all queries at once as the search function expects
        print("Searching with multiple queries...")
        results = search_materials_database(test_queries)
        
        if results:
            print(f"\nFound {len(results)} unique document segments:")
            for i, result in enumerate(results, 1):
                # Print truncated results for readability
                truncated = result[:300] + "..." if len(result) > 300 else result
                print(f"\nResult {i}:\n{truncated}")
                if len(result) > 300:
                    print("(content truncated for readability)")
        else:
            print("No results found.")
    
    except Exception as e:
        logger.error(f"Error testing search_materials_database: {str(e)}")
        print(f"ERROR: {str(e)}")
    
    print("\n" + "="*80)
    print("Test completed.")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_search_materials_database()