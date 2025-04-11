import os
import sys
from loguru import logger
from dotenv import load_dotenv

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data_loader.doc_loader import load_initial_data
from src.ai_functions.prompt_functions import sentence_transformer_embeddings

# Load environment variables
load_dotenv()

# Setup logging
logger.add("logs/indexing.log", rotation="500 MB")

def main():
    """Index all PDF files in the data directory"""
    try:
        logger.info("Starting data indexing process...")
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output/doc_indexes", exist_ok=True)
        
        # Load and index all PDF files
        indices = load_initial_data(sentence_transformer_embeddings)
        
        if indices:
            logger.success(f"Successfully indexed {len(indices)} files:")
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