import os
import sys
from typing import List

# Add the root project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from loguru import logger
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.data_loader import settings
from src.ai_functions.prompts import *
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize sentence transformer embeddings
sentence_transformer_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# LLAMA 3.1
llama_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY, 
    temperature=0.0,
    max_retries=5 
)

# QWEN LLM
qwen_llm = ChatGroq(
    model="qwen-2.5-32b",  
    api_key=GROQ_API_KEY, 
    temperature=0.0,
    max_retries=5 
)


def generate_multiple_queries(query: str, llm=llama_llm) -> list:
    """
    Generate multiple related queries from an initial query to explore a topic in depth.
    Returns JSON format with "queries" as the key containing an array of query strings.
    
    Args:
        query: Initial user query
        llm: The language model
        
    Returns:
        list: List of related queries
    """
    query_prompt = PromptTemplate(
        input_variables=["query"],
        template=multiple_query_prompt
    )
    query_generator = query_prompt | llm | JsonOutputParser()
    
    try:
        # Generate related queries as JSON
        result = query_generator.invoke({
            "query": query,
        })
        
        # Extract queries list from the JSON result
        related_queries = result.get("queries", [])
        
        # Ensure all queries end with question marks
        related_queries = [q if q.endswith('?') else f"{q}?" for q in related_queries]
        
        logger.info(f"Generated queries: {related_queries}")
        return related_queries
        
    except Exception as e:
        logger.error(f"Query generation failed with main model: {str(e)}")
        return [query]  # Return original query if generation fails


def analyze_document_content(query: str, retrieved_texts: List[str], llm=llama_llm) -> str:
    """
    Analyze and synthesize content from multiple retrieved document sections
    to provide a comprehensive response to the user's query.
    
    Args:
        query: The original user query
        retrieved_texts: List of text chunks retrieved from documents
        llm: The language model
        
    Returns:
        str: Comprehensive analysis and response to the query
    """
    # Join the retrieved texts with clear separators
    combined_texts = "\n\n---\n\n".join([f"Document segment {i+1}:\n{text}" 
                                        for i, text in enumerate(retrieved_texts)])
    
    # Create a prompt for comprehensive analysis
    analysis_prompt = PromptTemplate(
        input_variables=["query", "retrieved_texts"],
        template=comprehensive_response_prompt
    )
    
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    
    try:
        analysis = analysis_chain.invoke({
            "query": query,
            "retrieved_texts": combined_texts
        })
        return analysis
    except Exception as e:
        logger.error(f"Document analysis failed: {str(e)}")
        return "Error analyzing document content."
        
    

if __name__ == "__main__":
    query = "What is the material selection process for a man powered plane operation"
    queries = generate_multiple_queries(query=query)
    print(queries)