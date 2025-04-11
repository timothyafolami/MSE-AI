# AI-Driven Chatbot with Dual-Mode Functionality (Conversational + Material Science RAG)

## Overview

This project centers on building a smart chatbot application that has two primary modes of operation:

1. **Conversational Mode** – For everyday, friendly Q&A.
2. **Material Science Expert Mode (MSE-AI)** – A specialized Retrieval-Augmented Generation (RAG) system that assists with material selection and metallurgy-related questions.

The chatbot switches between modes based on user queries. It uses a vector database containing data from textbooks on material science and metallurgy to provide in-depth, evidence-based answers when questions fall under the Material Science Expert mode.

---

## Goals and Objectives

1. **Conversational Mode**

   - Provide user-friendly, natural, and coherent responses to general user queries.
   - Leverage a Large Language Model (LLM) (e.g., Llama) with an appropriate prompt template for natural language conversation.
2. **Material Science Expert Mode (MSE-AI)**

   - Offer in-depth technical support for queries related to materials science and metallurgy.
   - Retrieve relevant data from a vector database storing textbooks on materials and metallurgy.
   - Guide the user through a refined questioning process to gather all necessary details for accurate material recommendations.
   - Return the best material suggestion along with the reasoning behind it based on user inputs.

---

## Technical Flow

1. **User Query Check**

   - When a user sends a query:
     - If it is **conversational** (generic, non-technical question), the system calls the **General Response** prompt function to provide a natural, friendly LLM-generated answer.
     - If it is **material science–focused** (technical query related to materials and processes), the system switches to **MSE-AI Mode**.
2. **Material Science Expert Mode (MSE-AI)**When activated:

   1. **Query Analysis**

      - A “Query Reviewer” function analyzes the user’s initial question.
      - It generates **4 sub-questions** that aim to clarify the user’s needs, constraints, and any additional context (e.g., environment, budget, mechanical requirements).
      - These 4 generated questions are stored in the application’s session state.
   2. **Interactive User Q&A**

      - The system sequentially asks the user each sub-question **one at a time** (never all together).
      - Each user response is saved. If a user’s response is empty, “nil,” or “none,” the system will record a best possible assumption for that question.
   3. **Query Refining**

      - After the user answers all 4 questions, the system calls the “Query Refiner” (another LLM prompt function), providing:
        1. The original query
        2. The list of sub-questions
        3. The user’s responses to those sub-questions
      - The Query Refiner generates **another set of 4 refined questions**, now more focused and precise based on the user’s situation and new context.
   4. **Vector Database Lookup**

      - Using the original query, the sub-questions, user responses, and any refined questions, the system queries the vector database of materials and metallurgy references (textbook data).
      - Relevant passages are retrieved to build a context for the material selection process.
   5. **Material Selection**

      - The system then calls the “Material Selection” prompt function to:
        1. Interpret the user’s requirements from the refined context.
        2. Evaluate the retrieved passages from the vector database.
        3. Determine the most suitable material(s) for the user’s specific application or process.
   6. **Final Response**

      - The system returns:
        1. The **best-suited material** recommendation.
        2. An **explanation** justifying why this material is optimal, referencing the user’s inputs and the technical data retrieved.

---

## Prompt Functions

1. **Query Analysis** – Determines whether a user query is conversational or material science–focused and generates the initial 4 sub-questions if needed.
2. **General Response** – Provides a natural, conversational response for general queries.
3. **Question Generation** – Responsible for creating the sub-questions during the MSE-AI flow.
4. **Question Refining** – Generates new refined questions after the user has answered the initial sub-questions.
5. **Material Selection** – Interprets refined user context, queries the vector database, and selects the most suitable material with accompanying explanations.

---

## Technical Stack

1. **LLM**
   - Use Llama or a similar large language model as the backbone for generating responses and refining queries.
2. **Vector Database**
   - Stores and retrieves relevant textbook data on material science and metallurgy.
   - Provides context passages for the system to process.
3. **Frontend**
   - **Streamlit** for the user interface, enabling interactive chat and step-by-step question answering.
4. **Session State**
   - Maintains the conversation context and stores the sub-questions and user responses to ensure continuity in the MSE-AI flow.

---

## How It All Ties Together

1. User opens the Streamlit application.
2. User types a question.
3. The **Query Analysis** function checks if the question is general or about materials science.
4. For general queries, the **General Response** function is used to produce a direct, friendly answer.
5. For material science queries, the system enters **MSE-AI** mode:
   1. The system generates 4 sub-questions (saved in session state).
   2. The user is asked each question in turn, with responses recorded.
   3. The system refines the query using these responses.
   4. The vector database is queried for relevant information.
   5. The “Material Selection” function returns a final recommendation and rationale.

---

## Conclusion

This dual-mode chatbot project merges conversational AI with a specialized RAG workflow, offering both general Q&A and in-depth, data-driven material recommendations. By integrating Llama-based LLM prompts and a vector database of materials science resources, the system delivers accurate, context-aware insights. Streamlit’s simple and interactive interface makes this solution approachable for end-users seeking either everyday queries or expert-level materials counsel.
