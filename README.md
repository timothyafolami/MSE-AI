# Material Science Document Assistant

A Streamlit-based application for analyzing and querying material science documents using AI.

## Features

- Upload and process material science documents (PDF, DOCX, TXT)
- Query document content using natural language
- AI generates multiple related queries to explore topics in depth
- Vector database for efficient document retrieval
- Support for multiple document types and formats

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MSE-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file in the project root
- Add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Upload documents through the sidebar
3. Process the documents using the "Process Documents" button
4. Ask questions about your documents in the main interface

## Project Structure

```
MSE-AI/
├── data/                   # Sample data files
├── logs/                   # Application logs
├── output/                 # Generated indices and outputs
│   └── doc_indexes/        # Document vector indices
├── src/                    # Source code
│   ├── ai_functions/       # AI prompt functions
│   │   ├── prompt_functions.py
│   │   └── prompts.py
│   └── data_loader/        # Document processing modules
│       ├── doc_indexer.py
│       ├── doc_loader.py
│       ├── pdf_loader.py
│       ├── settings.py
│       └── unstructured_loader.py
├── temp_uploads/           # Temporary storage for uploaded files
├── .env                    # Environment variables
├── main.py                 # Main application entry point
└── requirements.txt        # Project dependencies
```

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API key or Groq API key
- FAISS for vector indexing