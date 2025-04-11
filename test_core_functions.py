import os
import sys
from loguru import logger
from dotenv import load_dotenv

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.ai_functions.prompt_functions import (
    determine_query_mode,
    generate_conversational_response,
    generate_initial_questions,
    generate_refined_questions,
    create_comprehensive_query,
    generate_material_recommendations,
    sentence_transformer_embeddings
)

# Load environment variables
load_dotenv()

# Setup logging
logger.add("logs/test.log", rotation="500 MB")

def test_conversational_mode():
    """Test the conversational mode functionality"""
    logger.info("Testing conversational mode...")
    
    # Test queries
    conversational_queries = [
        "What's the weather like today?",
        "Tell me a joke about science",
        "Who won the World Cup in 2018?",
        "What's your favorite color?"
    ]
    
    for query in conversational_queries:
        print(f"\n=== Testing query: '{query}' ===")
        
        # Determine mode
        mode = determine_query_mode(query)
        print(f"Detected mode: {mode}")
        
        if mode == "CONVERSATIONAL":
            # Generate response
            response = generate_conversational_response(query)
            print(f"Response: {response}")
        else:
            print(f"UNEXPECTED: Query was classified as {mode}")
    
    logger.info("Conversational mode testing completed")
    

def test_material_science_mode():
    """Test the material science expert mode functionality"""
    logger.info("Testing material science mode...")
    
    # Test query
    query = "I want to develop a lightweight drone frame that can withstand outdoor conditions."
    print(f"\n=== Testing query: '{query}' ===")
    
    # Determine mode
    mode = determine_query_mode(query)
    print(f"Detected mode: {mode}")
    
    if mode == "MATERIAL_SCIENCE":
        # Generate initial questions
        print("\n=== Initial Questions ===")
        initial_questions = generate_initial_questions(query)
        for i, question in enumerate(initial_questions):
            print(f"{i+1}. {question}")
        
        # Simulate answers to initial questions
        initial_answers = {
            initial_questions[0]: "The drone will operate in temperatures between -10°C to 40°C.",
            initial_questions[1]: "The frame needs to support a payload of 2kg while keeping the total weight under 1kg.",
            initial_questions[2]: "It will be used outdoors, exposed to rain, UV radiation, and occasional salt spray near coastal areas.",
            initial_questions[3]: "We're planning to use CNC machining initially, then injection molding for scale production."
        }
        
        # Generate refined questions
        print("\n=== Refined Questions ===")
        refined_questions = generate_refined_questions(query, initial_answers)
        for i, question in enumerate(refined_questions):
            print(f"{i+1}. {question}")
        
        # Simulate answers to refined questions
        refined_answers = {
            refined_questions[0]: "Maximum temperature spikes of 50°C in direct sunlight.",
            refined_questions[1]: "The frame should withstand impact forces of up to 5G without permanent deformation.",
            refined_questions[2]: "The material should have excellent UV resistance and not degrade over 3 years of outdoor exposure.",
            refined_questions[3]: "The budget for materials is $200 per frame, and we need to make about 50 units initially."
        }
        
        # Create comprehensive query
        print("\n=== Comprehensive Query ===")
        comprehensive_query = create_comprehensive_query(query, initial_answers, refined_answers)
        print(comprehensive_query)
        
        # Generate material recommendations
        print("\n=== Material Recommendations ===")
        recommendations = generate_material_recommendations(comprehensive_query)
        print(recommendations)
    else:
        print(f"UNEXPECTED: Query was classified as {mode}")
    
    logger.info("Material science mode testing completed")


def main():
    """Test both modes of the dual-mode chatbot"""
    try:
        logger.info("Starting dual-mode chatbot testing...")
        
        # Test conversational mode
        test_conversational_mode()
        
        # Test material science mode
        test_material_science_mode()
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        sys.exit(1)
        
    logger.info("All tests completed successfully.")


if __name__ == "__main__":
    # Create required directories
    os.makedirs("logs", exist_ok=True)
    main()