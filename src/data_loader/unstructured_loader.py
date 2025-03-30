import sys
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from loguru import logger
from src.data_loader.doc_indexer import create_and_save_document_index
from src.data_loader import settings

class DataIndexer:
    """
    Unified document indexer that handles PDF, DOCX/DOC, and TXT files.
    Creates FAISS indices using the specified embedding model.
    """
    
    def __init__(self,
                 embeddings, 
                 input_paths: List[str],
                 max_workers: int = 4):
        """
        Initialize the document indexer.
        
        Args:
            embeddings: The embeddings object to use
            input_paths (List[str]): List of file/directory paths to index
            max_workers (int): Maximum concurrent processing threads
        """
        self.embeddings = embeddings
        self.input_paths = input_paths
        self.max_workers = max_workers
        self.supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
        
        # Track processing results
        self.successful_indices = []
        self.failed_files = []

    def get_files_to_process(self) -> List[str]:
        """Collect all supported files from input paths"""
        files = []
        for path in self.input_paths:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                if Path(path).suffix.lower() in self.supported_extensions:
                    files.append(path)
            elif os.path.isdir(path):
                for root, _, filenames in os.walk(path):
                    for fname in filenames:
                        file_path = os.path.join(root, fname)
                        if Path(file_path).suffix.lower() in self.supported_extensions:
                            files.append(file_path)
        return files

    def process_file(self, file_path: str) -> Optional[str]:
        """Index a single file and return the index path"""
        try:
            index_path = create_and_save_document_index(embeddings=self.embeddings, document_path=file_path)
            logger.success(f"Indexed {file_path} → {index_path}")
            return index_path
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {str(e)}")
            return None

    def process_all_files(self) -> List[str]:
        """Process all files with concurrent execution"""
        files = self.get_files_to_process()
        if not files:
            logger.warning("No supported files found to index")
            return []

        logger.info(f"Indexing {len(files)} files with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_file, f): f for f in files}
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    index_path = future.result()
                    if index_path:
                        self.successful_indices.append(index_path)
                    else:
                        self.failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                    self.failed_files.append(file_path)

        return self.successful_indices

    def get_summary(self) -> dict:
        """Get processing summary statistics"""
        return {
            "total_files": len(self.successful_indices) + len(self.failed_files),
            "successful": {
                "count": len(self.successful_indices),
                "paths": self.successful_indices
            },
            "failed": {
                "count": len(self.failed_files),
                "files": self.failed_files
            }
        }