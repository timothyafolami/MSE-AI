import os
import sys
import argparse
import shutil
from loguru import logger
from dotenv import load_dotenv

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data_loader.doc_loader import load_initial_data
from src.ai_functions.prompt_functions import sentence_transformer_embeddings
from src.data_loader import settings

# Load environment variables
load_dotenv()

# Setup logging
logger.add("logs/indexing.log", rotation="500 MB")

def main():
    """Index all supported files in the data directory"""
    parser = argparse.ArgumentParser(description='Index documents for the materials engineering knowledge base.')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the index from scratch')
    args = parser.parse_args()
    
    try:
        logger.info("Starting data indexing process...")
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output/doc_indexes", exist_ok=True)
        
        # Handle rebuilding the index if requested
        if args.rebuild:
            unified_index_path = os.path.join(settings.DOC_INDEXES_DIR, "materials_database")
            if os.path.exists(unified_index_path):
                logger.info(f"Removing existing unified index at {unified_index_path}")
                shutil.rmtree(unified_index_path, ignore_errors=True)
                logger.success("Existing index removed for rebuilding")
            else:
                logger.info("No existing index found to rebuild")
        
        # Load and index all supported files
        indices = load_initial_data(sentence_transformer_embeddings)
        
        if indices:
            logger.success(f"Successfully indexed documents into unified database:")
            for idx_path in indices:
                logger.info(f"  - {idx_path}")
        else:
            logger.warning("No files were indexed. Please check the data directory.")
            
    except Exception as e:
        logger.error(f"Error during indexing process: {str(e)}")
        sys.exit(1)
        
    logger.info("Indexing process completed.")

if __name__ == "__main__":
    main()