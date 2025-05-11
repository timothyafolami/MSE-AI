import os
import sys
import streamlit as st
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Material Selection Professional Assistant",
    page_icon="🧪",
    layout="wide"
)

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Setup logging
logger.add("logs/app.log", rotation="500 MB")

def main():
    """Main function for the home page"""
    # Initialize session state variables for the home page
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Center container for main content
    with st.container():
        # Main title
        st.title("Material Selection Professional Assistant")

        # Logos side by side
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image("images/oau_logo.jpeg", width=150)
            except Exception:
                st.write("OAU Logo")

        with col2:
            try:
                st.image("images/mse_logo.jpeg", width=150)
            except Exception:
                st.write("MSE Logo")

        # Developer information
        st.markdown("### Developed By:")
        st.markdown("- **Supervisor**: Dr. Daniyan")
        st.markdown("- **Student**: Okegbenro Stephen")

        # About the project (shorter and more concise)
        st.markdown("### About This Project")
        st.markdown("This project is conducted by the Department of Material Science and Engineering, OAU. It leverages artificial intelligence to assist in selecting appropriate materials for various engineering applications.")

        # Key features in a more compact format
        st.markdown("### Key Features:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- Material Selection Guidance")
            st.markdown("- Educational Support")
        with col2:
            st.markdown("- Research Assistant")
            st.markdown("- Expert Recommendations")

        # How it works - more compact
        st.markdown("### How It Works:")
        st.markdown("1. Navigate to the Chat page")
        st.markdown("2. Describe your material requirements")
        st.markdown("3. Answer the follow-up questions")
        st.markdown("4. Receive tailored material recommendations")

        # Navigation button to chat page
        st.markdown("### Ready to start?")
        if st.button("Go to Chat Interface", use_container_width=True):
            # Navigate to the chat page
            st.switch_page("pages/01_Chat.py")

if __name__ == "__main__":
    main()