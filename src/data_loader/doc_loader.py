import os
import docx
from typing import Dict, List
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from loguru import logger

# Import the PDF loading functions from pdf_loader.py
from src.data_loader.pdf_loader import load_pdf  

class DataLoader:
    """
    Loads text data from multiple files. Returns a dictionary with file paths 
    as keys and the extracted text as values.
    """

    def __init__(self, document_paths: List[str]):
        """
        Args:
            document_paths (List[str]): A list of file or directory paths.
        """
        self.document_paths = document_paths
        self.supported_extensions = ['.pdf', '.docx', '.doc', '.txt']

    def _get_files_to_process(self) -> List[str]:
        """
        Collect all supported files from the given list of paths.
        
        Returns:
            List[str]: A list of absolute file paths matching the supported extensions.
        """
        files = []
        for path in self.document_paths:
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

    def _extract_text(self, file_path: str):
        """
        Extract text from a single file based on its extension.

        Args:
            file_path (str): The absolute path to the file.

        Returns:
            Documents or str: The extracted text or document objects from the file.
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            try:
                documents = load_pdf(file_path)
                return documents
            except Exception as e:
                logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
                return None

        elif ext in ['.docx', '.doc']:
            try:
                doc = docx.Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                text = "\n\n".join(paragraphs)
                return text
            except Exception as e:
                logger.error(f"Word document extraction failed for {file_path}: {str(e)}")
                return None

        elif ext == '.txt':
            try:
                loader = TextLoader(file_path)
                documents = loader.load()
                return documents[0].page_content if documents else None
            except Exception as e:
                logger.error(f"Text file extraction failed for {file_path}: {str(e)}")
                return None

        # If extension not supported or something goes wrong
        return None

    def load_data(self) -> Dict[str, str]:
        """
        Main method for loading text data from the provided document paths.

        Returns:
            Dict[str, str]: A dictionary mapping file paths to their extracted text.
        """
        data_dict = {}
        files_to_process = self._get_files_to_process()

        if not files_to_process:
            logger.warning("No supported files found for data loading.")
            return data_dict

        logger.info(f"Loading text data from {len(files_to_process)} files...")

        for file_path in files_to_process:
            try:
                text_content = self._extract_text(file_path)
                if text_content:
                    data_dict[file_path] = text_content
                    logger.success(f"Successfully loaded: {file_path}")
                else:
                    logger.warning(f"No content extracted from: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load data from {file_path}: {str(e)}")

        return data_dict