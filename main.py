import os
import sys
import streamlit as st
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data_loader.doc_loader import load_initial_data
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
logger.add("logs/app.log", rotation="500 MB")

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'mode' not in st.session_state:
        st.session_state.mode = None  # Will be set to "CONVERSATIONAL" or "MATERIAL_SCIENCE"
    
    # Initial MSE-AI flow state variables
    if 'initial_questions' not in st.session_state:
        st.session_state.initial_questions = []
    
    if 'initial_qa' not in st.session_state:
        st.session_state.initial_qa = {}
    
    if 'initial_questions_answered' not in st.session_state:
        st.session_state.initial_questions_answered = False
    
    # Refined MSE-AI flow state variables
    if 'refined_questions' not in st.session_state:
        st.session_state.refined_questions = []
    
    if 'refined_qa' not in st.session_state:
        st.session_state.refined_qa = {}
    
    if 'refined_questions_answered' not in st.session_state:
        st.session_state.refined_questions_answered = False
    
    # User's original query for MSE-AI mode
    if 'original_query' not in st.session_state:
        st.session_state.original_query = ""
    
    # Material recommendations state
    if 'recommendation_provided' not in st.session_state:
        st.session_state.recommendation_provided = False
        
    # Conversation history storage
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        
    # Current conversation ID/timestamp
    if 'current_conversation_id' not in st.session_state:
        from datetime import datetime
        st.session_state.current_conversation_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def reset_session_state():
    """Reset the session state for a new conversation"""
    # Save current conversation to history if it's not empty
    if 'conversation' in st.session_state and st.session_state.conversation:
        # Create a conversation entry with timestamp and messages
        conversation_entry = {
            "id": st.session_state.current_conversation_id,
            "messages": st.session_state.conversation.copy(),
            "mode": st.session_state.mode
        }
        
        # Add to history
        st.session_state.conversation_history.append(conversation_entry)
        
        # Limit history to most recent 10 conversations
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
    
    # Create new conversation ID
    from datetime import datetime
    st.session_state.current_conversation_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reset conversation state
    if 'conversation' in st.session_state:
        st.session_state.conversation = []
    if 'mode' in st.session_state:
        st.session_state.mode = None
    if 'initial_questions' in st.session_state:
        st.session_state.initial_questions = []
    if 'initial_qa' in st.session_state:
        st.session_state.initial_qa = {}
    if 'initial_questions_answered' in st.session_state:
        st.session_state.initial_questions_answered = False
    if 'refined_questions' in st.session_state:
        st.session_state.refined_questions = []
    if 'refined_qa' in st.session_state:
        st.session_state.refined_qa = {}
    if 'refined_questions_answered' in st.session_state:
        st.session_state.refined_questions_answered = False
    if 'original_query' in st.session_state:
        st.session_state.original_query = ""
    if 'recommendation_provided' in st.session_state:
        st.session_state.recommendation_provided = False


def get_conversation_summary(conversation):
    """Return a brief summary of the conversation based on the first user message"""
    if conversation and "messages" in conversation and conversation["messages"]:
        for message in conversation["messages"]:
            if message["role"] == "user":
                # Use the first 30 characters of the first user message as a summary
                summary = message["content"][:30]
                if len(message["content"]) > 30:
                    summary += "..."
                return summary
    return "Empty conversation"


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Material Selection Professional Assistant",
        page_icon="🧪",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()

    # Create sidebar
    with st.sidebar:
        # OAU Logo
        st.image("/Users/apple/Downloads/MSE-AI/images/oau_logo.jpeg", width=150)
        
        st.markdown("### This project is conducted by the Department of Material Science and Engineering, OAU")
        
        # MSE Logo
        st.image("/Users/apple/Downloads/MSE-AI/images/mse_logo.jpeg", width=150)
        
        st.markdown("**Supervisor Name:** Dr. John Smith")
        st.markdown("**Student Name:** Jane Doe")
        
        st.markdown("---")
        
        # Add reset button to sidebar
        if st.button("Start New Conversation"):
            reset_session_state()
            st.rerun()
            
        # Previous conversations section
        st.markdown("### Conversation History")
        
        # Check if we have previous conversations
        if st.session_state.conversation_history:
            # Reverse the list to show newest first
            history = st.session_state.conversation_history.copy()
            history.reverse()
            
            # Create a selectbox for conversation history
            conversation_options = [f"{conv['id']} - {get_conversation_summary(conv)}" for conv in history]
            selected_conversation = st.selectbox("Previous conversations:", 
                                               conversation_options,
                                               key="conversation_selector")
            
            if st.button("Load Selected Conversation"):
                # Find the selected conversation
                selected_id = selected_conversation.split(" - ")[0]
                for conv in st.session_state.conversation_history:
                    if conv['id'] == selected_id:
                        # Load this conversation
                        st.session_state.conversation = conv['messages'].copy()
                        st.session_state.mode = conv['mode']
                        st.rerun()
        else:
            st.markdown("No previous conversations yet.")
    
    # Page title and description
    st.title("Material Selection Professional Assistant")
    st.write("Your expert guide for material selection and recommendations based on engineering requirements.")
    
    # Display conversation history
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_input = st.chat_input("What would you like to ask or discuss?")
    
    if user_input:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # If this is a new query (no mode set yet)
        if st.session_state.mode is None:
            with st.spinner("Analyzing your query..."):
                # Determine if the query is conversational or materials science focused
                mode = determine_query_mode(user_input)
                st.session_state.mode = mode
                logger.info(f"Query mode determined: {mode}")
                
                if mode == "CONVERSATIONAL":
                    # Generate a conversational response
                    response = generate_conversational_response(user_input)
                    
                    st.session_state.conversation.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
                
                else:  # MATERIAL_SCIENCE mode
                    # Store the original query
                    st.session_state.original_query = user_input
                    
                    # Generate initial questions
                    questions = generate_initial_questions(user_input)
                    st.session_state.initial_questions = questions
                    
                    # Display the first question
                    response = "Thank you for your materials science question. To provide the best recommendation, I'll need some additional information. Let's start with:"
                    response += f"\n\n{questions[0]}"
                    
                    st.session_state.conversation.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
        
        # Handle follow-up responses in MATERIAL_SCIENCE mode - initial questions
        elif st.session_state.mode == "MATERIAL_SCIENCE" and not st.session_state.initial_questions_answered:
            # Store the user's answer to the current question
            current_question_index = len(st.session_state.initial_qa)
            current_question = st.session_state.initial_questions[current_question_index]
            
            # Handle empty answers or "none"/"nil" with a default assumption
            if user_input.lower() in ["none", "nil", ""] or user_input.isspace():
                user_input = "No specific requirement provided. Please make a best assumption."
            
            st.session_state.initial_qa[current_question] = user_input
            
            # Check if we have more initial questions to ask
            if current_question_index + 1 < len(st.session_state.initial_questions):
                # Ask the next question
                next_question = st.session_state.initial_questions[current_question_index + 1]
                response = f"Thank you. Next question:\n\n{next_question}"
                
                st.session_state.conversation.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            else:
                # All initial questions answered, generate refined questions
                st.session_state.initial_questions_answered = True
                
                with st.spinner("Analyzing your responses and generating follow-up questions..."):
                    # Generate refined questions
                    refined_questions = generate_refined_questions(
                        st.session_state.original_query,
                        st.session_state.initial_qa
                    )
                    st.session_state.refined_questions = refined_questions
                    
                    # Ask the first refined question
                    response = "Great! Based on your answers, I have a few more specific questions to better understand your requirements:"
                    response += f"\n\n{refined_questions[0]}"
                    
                    st.session_state.conversation.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
        
        # Handle follow-up responses in MATERIAL_SCIENCE mode - refined questions
        elif st.session_state.mode == "MATERIAL_SCIENCE" and st.session_state.initial_questions_answered and not st.session_state.refined_questions_answered:
            # Store the user's answer to the current refined question
            current_question_index = len(st.session_state.refined_qa)
            current_question = st.session_state.refined_questions[current_question_index]
            
            # Handle empty answers or "none"/"nil" with a default assumption
            if user_input.lower() in ["none", "nil", ""] or user_input.isspace():
                user_input = "No specific requirement provided. Please make a best assumption."
            
            st.session_state.refined_qa[current_question] = user_input
            
            # Check if we have more refined questions to ask
            if current_question_index + 1 < len(st.session_state.refined_questions):
                # Ask the next refined question
                next_question = st.session_state.refined_questions[current_question_index + 1]
                response = f"Thank you. Next question:\n\n{next_question}"
                
                st.session_state.conversation.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            else:
                # All refined questions answered, generate material recommendations
                st.session_state.refined_questions_answered = True
                
                with st.spinner("Analyzing your requirements and searching for optimal materials..."):
                    # Create comprehensive query from all answers
                    comprehensive_query = create_comprehensive_query(
                        st.session_state.original_query,
                        st.session_state.initial_qa,
                        st.session_state.refined_qa
                    )
                    
                    # Generate material recommendations
                    recommendations = generate_material_recommendations(comprehensive_query)
                    st.session_state.recommendation_provided = True
                    
                    response = "Based on your requirements, here are my material recommendations:\n\n"
                    response += recommendations
                    response += "\n\nYou can ask follow-up questions about these materials or start a new inquiry at any time."
                    
                    st.session_state.conversation.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
        
        # Handle follow-up questions after recommendations are provided or in conversational mode
        elif (st.session_state.mode == "MATERIAL_SCIENCE" and st.session_state.recommendation_provided) or st.session_state.mode == "CONVERSATIONAL":
            with st.spinner("Processing your question..."):
                # Check if we should switch modes for this follow-up question
                new_mode = determine_query_mode(user_input)
                
                if new_mode != st.session_state.mode:
                    # Mode has changed, reset the flow
                    reset_session_state()
                    st.session_state.mode = new_mode
                    
                    if new_mode == "CONVERSATIONAL":
                        # Generate a conversational response
                        response = generate_conversational_response(user_input)
                        
                        st.session_state.conversation.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.write(response)
                    else:  # Switched to MATERIAL_SCIENCE mode
                        # Store the original query
                        st.session_state.original_query = user_input
                        
                        # Generate initial questions
                        questions = generate_initial_questions(user_input)
                        st.session_state.initial_questions = questions
                        
                        # Display the first question
                        response = "Let me help with your materials science question. To provide the best recommendation, I'll need some additional information. Let's start with:"
                        response += f"\n\n{questions[0]}"
                        
                        st.session_state.conversation.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.write(response)
                else:
                    # Same mode, treat as a follow-up question
                    if st.session_state.mode == "CONVERSATIONAL":
                        response = generate_conversational_response(user_input)
                    else:  # MATERIAL_SCIENCE follow-up
                        # For follow-up in material science mode, we'll treat it as a new conversation related to materials
                        # A more sophisticated approach would use RAG to generate a response based on the vector database
                        response = generate_conversational_response(user_input)
                    
                    st.session_state.conversation.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)

if __name__ == "__main__":
    main()