import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Base directories
ROOT_DIR = str(PROJECT_ROOT)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
TEMP_DIR = os.path.join(ROOT_DIR, "temp_uploads")

# Output subdirectories
DOC_INDEXES_DIR = os.path.join(OUTPUT_DIR, "doc_indexes")

# Create directories if they don't exist
for directory in [
    OUTPUT_DIR,
    LOGS_DIR,
    DOC_INDEXES_DIR,
    TEMP_DIR,
]:
    os.makedirs(directory, exist_ok=True)