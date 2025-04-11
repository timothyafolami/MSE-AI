import os
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Union
import json

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
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_loader.doc_indexer import retrieve_documents

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

# QWEN LLM for conversational mode (higher temperature)
conv_llm = ChatGroq(
    model="qwen-2.5-32b",  
    api_key=GROQ_API_KEY, 
    temperature=0.7,  # Higher temperature for more creative responses
    max_retries=5 
)


def determine_query_mode(query: str, llm=llama_llm) -> str:
    """
    Determine if a query is conversational or materials science focused.
    
    Args:
        query: User's input query
        llm: The language model
        
    Returns:
        str: Either "CONVERSATIONAL" or "MATERIAL_SCIENCE"
    """
    mode_prompt = PromptTemplate(
        input_variables=["query"],
        template=query_analysis_prompt
    )
    
    mode_chain = mode_prompt | llm | StrOutputParser()
    
    try:
        mode = mode_chain.invoke({"query": query}).strip()
        logger.info(f"Query mode determined: {mode}")
        
        # Ensure the mode is one of the expected values
        if mode not in ["CONVERSATIONAL", "MATERIAL_SCIENCE"]:
            logger.warning(f"Unexpected mode returned: {mode}. Defaulting to CONVERSATIONAL")
            return "CONVERSATIONAL"
        
        return mode
    except Exception as e:
        logger.error(f"Mode determination failed: {str(e)}")
        # Default to conversational mode if determination fails
        return "CONVERSATIONAL"


def generate_conversational_response(query: str, llm=conv_llm) -> str:
    """
    Generate a conversational response to the user's query.
    
    Args:
        query: User's query
        llm: The language model
        
    Returns:
        str: Natural language response
    """
    response_prompt = PromptTemplate(
        input_variables=["query"],
        template=general_response_prompt
    )
    
    response_chain = response_prompt | llm | StrOutputParser()
    
    try:
        response = response_chain.invoke({"query": query})
        logger.info("Generated conversational response")
        return response
    except Exception as e:
        logger.error(f"Conversational response generation failed: {str(e)}")
        return "I'm sorry, I'm having trouble formulating a response right now. Could you try phrasing your question differently?"


def generate_initial_questions(query: str, llm=llama_llm) -> list:
    """
    Generate 4 initial sub-questions to gather requirements for material selection.
    
    Args:
        query: User's query
        llm: The language model
        
    Returns:
        list: List of 4 questions
    """
    question_prompt = PromptTemplate(
        input_variables=["query"],
        template=initial_questions_prompt
    )
    question_generator = question_prompt | llm | JsonOutputParser()
    
    try:
        # Generate questions as JSON
        result = question_generator.invoke({
            "query": query,
        })        
        # Extract questions list from the JSON result
        questions = result.get("questions", [])
        
        logger.info(f"Generated {len(questions)} initial questions")
        return questions
        
    except Exception as e:
        logger.error(f"Initial question generation failed: {str(e)}")
        # Return some default questions if generation fails
        return [
            "What is the maximum operating temperature the material will be exposed to?",
            "What strength requirements does your application have?",
            "What environmental conditions will the material be exposed to?",
            "What manufacturing process do you plan to use?"
        ]


def generate_refined_questions(original_query: str, question_answers: Dict[str, str], llm=llama_llm) -> list:
    """
    Generate refined follow-up questions based on initial responses.
    
    Args:
        original_query: Original user query
        question_answers: Dictionary of initial questions and user answers
        llm: The language model
        
    Returns:
        list: List of 4 refined questions
    """
    # Format the question-answer pairs for the prompt
    qa_formatted = "\n".join([f"Q: {q}\nA: {a}" for q, a in question_answers.items()])
    
    refiner_prompt = PromptTemplate(
        input_variables=["original_query", "question_answers"],
        template=question_refiner_prompt
    )
    question_refiner = refiner_prompt | llm | JsonOutputParser()
    
    try:
        # Generate refined questions as JSON
        result = question_refiner.invoke({
            "original_query": original_query,
            "question_answers": qa_formatted
        })
        
        # Extract questions list from the JSON result
        questions = result.get("questions", [])
        
        logger.info(f"Generated {len(questions)} refined questions")
        return questions
        
    except Exception as e:
        logger.error(f"Refined question generation failed: {str(e)}")
        # Return some default refined questions if generation fails
        return [
            "Can you provide more specific details about the performance requirements?",
            "Are there any specific material properties that are critical for your application?",
            "What is your budget range for the materials?",
            "Are there any specific materials you've considered or would like to avoid?"
        ]


def create_comprehensive_query(
    original_query: str, 
    initial_qa: Dict[str, str], 
    refined_qa: Dict[str, str], 
    llm=llama_llm
) -> str:
    """
    Process all question answers to create a comprehensive query.
    
    Args:
        original_query: Original user query
        initial_qa: Dictionary of initial questions and answers
        refined_qa: Dictionary of refined questions and answers
        llm: The language model
        
    Returns:
        str: Comprehensive query for material selection
    """
    # Format the question-answer pairs
    initial_qa_formatted = "\n".join([f"Q: {q}\nA: {a}" for q, a in initial_qa.items()])
    refined_qa_formatted = "\n".join([f"Q: {q}\nA: {a}" for q, a in refined_qa.items()])
    
    process_prompt = PromptTemplate(
        input_variables=["original_query", "initial_qa", "refined_qa"],
        template=process_answers_prompt
    )
    
    process_chain = process_prompt | llm | StrOutputParser()
    
    try:
        comprehensive_query = process_chain.invoke({
            "original_query": original_query,
            "initial_qa": initial_qa_formatted,
            "refined_qa": refined_qa_formatted
        })
        
        logger.info("Generated comprehensive query")
        return comprehensive_query
    except Exception as e:
        logger.error(f"Processing answers failed: {str(e)}")
        # Return a simplified version if processing fails
        return f"Query: {original_query}. Initial Specifications: {initial_qa_formatted}. Refined Specifications: {refined_qa_formatted}"


def generate_sub_queries(comprehensive_query: str, llm=llama_llm) -> List[str]:
    """
    Generate sub-queries for material selection based on the comprehensive query.
    
    Args:
        comprehensive_query: The processed comprehensive query
        llm: The language model
        
    Returns:
        List[str]: List of 4 targeted sub-queries
    """
    subquery_prompt = PromptTemplate(
        input_variables=["comprehensive_query"],
        template=material_search_prompt
    )
    
    subquery_chain = subquery_prompt | llm | StrOutputParser()
    
    try:
        result = subquery_chain.invoke({
            "comprehensive_query": comprehensive_query
        })
        
        # Extract just the sub-queries from the result
        sub_queries = []
        in_subqueries_section = False
        for line in result.split('\n'):
            if line.startswith('## Sub-Queries'):
                in_subqueries_section = True
                continue
            
            if in_subqueries_section and line.strip() and line[0].isdigit():
                # Extract just the query part (before the explanation)
                query_parts = line.split('-', 1)
                if len(query_parts) > 1:
                    # Remove the number and just get the query text
                    query = query_parts[0].split('.', 1)[1].strip()
                    sub_queries.append(query)
                else:
                    # If there's no explanation part, use the whole line
                    sub_queries.append(line.strip())
        
        logger.info(f"Generated {len(sub_queries)} sub-queries")
        return sub_queries
        
    except Exception as e:
        logger.error(f"Sub-query generation failed: {str(e)}")
        # Return generic sub-queries if generation fails
        return [
            "Materials with high strength-to-weight ratio",
            "Materials suitable for high temperature applications",
            "Materials with excellent corrosion resistance",
            "Materials suitable for standard manufacturing processes"
        ]


def search_materials_database(sub_queries: List[str], available_indices: List[str]) -> List[str]:
    """
    Search the materials database using the sub-queries.
    
    Args:
        sub_queries: List of targeted sub-queries
        available_indices: List of available document indices
        
    Returns:
        List[str]: Retrieved text segments
    """
    all_results = []
    
    # Process each document index
    for index_path in available_indices:
        try:
            document_name = Path(index_path).name
            logger.info(f"Searching in document: {document_name}")
            
            # Retrieve documents for each sub-query
            for query in sub_queries:
                try:
                    doc_results = retrieve_documents(
                        embeddings=sentence_transformer_embeddings,
                        query=query,
                        document_name=document_name,
                        search_type="mmr",
                        k=3  # Limit results per query to avoid too much data
                    )
                    all_results.extend([d.page_content for d in doc_results])
                    logger.info(f"Retrieved {len(doc_results)} results for query: {query}")
                except Exception as e:
                    logger.error(f"Error retrieving documents for query '{query}': {str(e)}")
        except Exception as e:
            logger.error(f"Error processing document index {index_path}: {str(e)}")
    
    # Remove duplicates while preserving order
    unique_results = []
    seen = set()
    for result in all_results:
        if result not in seen:
            seen.add(result)
            unique_results.append(result)
    
    logger.info(f"Retrieved {len(unique_results)} unique document segments")
    return unique_results


def generate_material_recommendations(comprehensive_query: str, llm=llama_llm) -> str:
    """
    Generate material recommendations based on comprehensive query.
    
    Args:
        comprehensive_query: The comprehensive query
        llm: The language model
        
    Returns:
        str: Material recommendations
    """
    try:
        # First, generate sub-queries for the search
        sub_queries = generate_sub_queries(comprehensive_query)
        
        # Get a list of all available document indices
        index_dirs = []
        for item in os.listdir(settings.DOC_INDEXES_DIR):
            full_path = os.path.join(settings.DOC_INDEXES_DIR, item)
            if os.path.isdir(full_path):
                index_dirs.append(full_path)
        
        # Search the materials database using the sub-queries
        retrieved_texts = search_materials_database(sub_queries, index_dirs)
        
        if not retrieved_texts:
            logger.warning("No relevant document segments found in the database")
            return "I couldn't find specific materials in our database that match your requirements. Please consider adjusting your specifications or consult with a materials specialist for custom recommendations."
        
        # Format the sub-queries for the prompt
        formatted_sub_queries = "\n".join([f"- {q}" for q in sub_queries])
        
        # IMPORTANT: Limit number of document segments to avoid token limits
        # Take only the most relevant documents (first 5)
        if len(retrieved_texts) > 5:
            logger.warning(f"Truncating {len(retrieved_texts)} retrieved texts to 5 to avoid token limits")
            retrieved_texts = retrieved_texts[:5]
        
        # Limit the length of each text segment (truncate to 1000 chars)
        truncated_texts = []
        for text in retrieved_texts:
            if len(text) > 1000:
                truncated_text = text[:1000] + "... [truncated]"
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)
        
        # Create the analysis prompt
        analysis_prompt = PromptTemplate(
            input_variables=["comprehensive_query", "sub_queries", "retrieved_texts"],
            template=material_analysis_prompt
        )
        
        analysis_chain = analysis_prompt | llm | StrOutputParser()
        
        # Generate material recommendations
        result = analysis_chain.invoke({
            "comprehensive_query": comprehensive_query,
            "sub_queries": formatted_sub_queries,
            "retrieved_texts": "\n\n---\n\n".join([f"Segment {i+1}:\n{text}" for i, text in enumerate(truncated_texts)])
        })
        
        logger.success("Successfully generated material recommendations")
        return result
        
    except Exception as e:
        logger.error(f"Material recommendation generation failed: {str(e)}")
        return "I encountered an error while generating material recommendations. Please try again with more specific requirements."


# Legacy function for backward compatibility
def generate_questions(project_description: str, llm=llama_llm) -> list:
    """Legacy function that calls generate_initial_questions"""
    return generate_initial_questions(project_description, llm)


# Legacy function for backward compatibility
def process_answers(project_description: str, question_answers: Dict[str, str], llm=llama_llm) -> str:
    """Legacy function for backward compatibility"""
    # This function now just creates a placeholder for refined QA
    empty_refined_qa = {}
    return create_comprehensive_query(project_description, question_answers, empty_refined_qa, llm)


if __name__ == "__main__":
    # Example usage
    query = "I want to build a lightweight bicycle frame that can withstand harsh weather conditions"
    mode = determine_query_mode(query)
    print(f"Query mode: {mode}")
    
    if mode == "CONVERSATIONAL":
        response = generate_conversational_response(query)
        print(f"Conversational response: {response}")
    else:
        questions = generate_initial_questions(query)
        print("Initial questions:")
        for q in questions:
            print(f"- {q}")
            
        # Simulate answers
        answers = {
            questions[0]: "The bike will be used in temperatures from -10°C to 40°C.",
            questions[1]: "It needs to support a rider weight of up to 100kg.",
            questions[2]: "It will be exposed to rain, sun, and occasionally salt spray near the coast.",
            questions[3]: "I'll use tube cutting and welding initially."
        }
        
        refined_questions = generate_refined_questions(query, answers)
        print("\nRefined questions:")
        for q in refined_questions:
            print(f"- {q}")
            
        # Simulate refined answers
        refined_answers = {
            refined_questions[0]: "Maximum temperature spikes could be up to 50°C in direct sunlight.",
            refined_questions[1]: "Strength to weight ratio is the most important factor.",
            refined_questions[2]: "It will also be exposed to road grime and occasional cleaning solvents.",
            refined_questions[3]: "Budget is up to $500 for the frame materials."
        }
        
        comprehensive_query = create_comprehensive_query(query, answers, refined_answers)
        print(f"\nComprehensive query:\n{comprehensive_query}")
        
        recommendations = generate_material_recommendations(comprehensive_query)
        print(f"\nRecommendations:\n{recommendations}")