import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag_project_impl import VectorStore, EmbeddingManager, RAGRetriever, rag_simple, initialize_vector_store_if_empty

# Load environment variables from the .env file
load_dotenv()

st.title("Cooking Assistant")
st.write("What do you want to cook")

# Custom CSS for Gemini-style layout (Right-aligned user, left-aligned bot, no avatars)
st.markdown("""
    <style>
    /* 1. Completely hide user and assistant default avatars/stickers */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    div[data-testid*="Avatar"] {
        display: none !important;
    }
    
    /* 2. Right-align the user chat message row */
    div.stChatMessage:has(div[data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse !important;
        text-align: right !important;
    }
    
    /* 3. Style the user message bubble and push it to the right */
    div.stChatMessage:has(div[data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        background-color: #2b313e !important;
        color: #ffffff !important;
        padding: 10px 14px;
        border-radius: 15px 15px 0 15px;
        display: inline-block;
        text-align: left;
        margin-left: auto !important;
        max-width: 75%;
    }

    /* 4. Left-align the assistant chat message row */
    div.stChatMessage:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row !important;
        text-align: left !important;
    }

    /* 3-dot thinking animation */
    @keyframes blink {
        0% { opacity: 0.2; }
        20% { opacity: 1; }
        100% { opacity: 0.2; }
    }
    .thinking-dot {
        animation: blink 1.4s infinite both;
        font-size: 28px;
        font-weight: bold;
        line-height: 10px;
    }
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
""", unsafe_allow_html=True)

# Load the RAG system components once so they stay cached in memory
@st.cache_resource
def load_rag_system():
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vectorstore = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
    initialize_vector_store_if_empty(vectorstore, embedding_manager, pdf_directory="data/pdf_documents")
    retriever = RAGRetriever(vectorstore, embedding_manager)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in the .env file.")
        return None, None
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=4096
    )
    return retriever, llm

retriever, llm = load_rag_system()

if retriever and llm:
    # Initialize chat history list in session memory if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all past chat messages on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box at the bottom of the page
    if prompt := st.chat_input("Eat Healthy"):
        
        # Save and display user message on the right
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from RAG pipeline on the left with thinking animation
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show 3-dot thinking animation
            message_placeholder.markdown(
                '<div style="display: flex; gap: 4px; align-items: center;">'
                '<span class="thinking-dot">.</span>'
                '<span class="thinking-dot">.</span>'
                '<span class="thinking-dot">.</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Fetch the actual answer from your RAG pipeline
            answer = rag_simple(prompt, retriever, llm, top_k=3)
            
            # Replace the 3 dots with the final response
            message_placeholder.markdown(answer)
                
        # Save assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})