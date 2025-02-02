import streamlit as st
import asyncio
from typing import List
import numpy as np
from typing import List, Dict, Any
from rag import hybrid_search
from result_format import YAMLResponse
from define_params import SessionManager, create_chain
from data_loading import doc_list, vectordb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
import time
import os

google_api_key = "AIzaSyAw4gUH6WEbGZaBg6KSLoYfU_djx-mEjxY"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# Initialize states
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = None
if 'typing_index' not in st.session_state:
    st.session_state.typing_index = 0
if 'last_type_time' not in st.session_state:
    st.session_state.last_type_time = time.time()
if 'yaml_chain_with_memory' not in st.session_state or 'fallback_chain_with_memory' not in st.session_state:
    st.session_state.yaml_chain_with_memory, st.session_state.fallback_chain_with_memory = create_chain(st.session_state.session_manager)

# Page config
st.set_page_config(
    page_title="Yamlify",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
/* Common styles */
[data-testid="stAppViewContainer"] {
    background-color: #140e2d;
    color: white;
}

/* Landing page specific */
.landing-container {
    display: flex;
    height: 100vh;
    padding: 2rem;
    gap: 100px;
    position: relative;
}

.left-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 16px;
}

.gradient-text {
    font-size: 128px;
    font-weight: bold;
    background: linear-gradient(to right, #217bfe, #e55571);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.start-button {
    padding: 15px 25px;
    background-color: #217bfe;
    color: white;
    border-radius: 20px;
    font-size: 14px;
    text-decoration: none;
    margin-top: 20px;
    transition: all 0.3s ease;
}

.start-button:hover {
    background-color: white;
    color: #217bfe;
}

.right-section {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.chat-box {
    background-color: #2c2937;
    padding: 20px;
    border-radius: 10px;
    position: absolute;
    bottom: -30px;
    right: -50px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.chat-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    object-fit: cover;
}

.bot-container {
    background-color: #140e2d;
    border-radius: 50px;
    width: 80%;
    height: 50%;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}

.bot-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.2;
    animation: slideBg 8s ease-in-out infinite alternate;
}

.bot-image {
    width: 100%;
    height: 100%;
    object-fit: contain;
    animation: botAnimate 3s ease-in-out infinite alternate;
            border-radius:80px;
}

.orbital {
    position: absolute;
    bottom: 0;
    left: 0;
    opacity: 0.05;
    animation: rotateOrbital 100s linear infinite;
    z-index: -1;
}

.terms {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
}

/* Dashboard specific styles */
.main {
    background-color: #1a1b26;
    color: #a9b1d6;
}

.stTextArea textarea {
    background-color: #24283b;
    color: #a9b1d6;
    border-radius: 10px;
    border: 1px solid #414868;
}

.stButton>button {
    background-color: #175ef7;
    color: white;
    border-radius: 8px;
    padding: 10px 25px;
    border: none;
    width: 100%;
}

.stButton>button:hover {
    background-color: #3d59a1;
}

.sidebar .sidebar-content {
    background-color: #16161e;
}

.css-1d391kg {
    background-color: #16161e;
}

.stMarkdown {
    color: #a9b1d6;
}

.css-1fv8s86 {
    background-color: #24283b;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

h1 {
    color: #ffffff !important;
}

h2 {
    color: #ffffff !important;
}

h3 {
    color: #ffffff !important;
}

code {
    background-color: #1a1b26 !important;
    padding: 10px !important;
    border-radius: 5px !important;
}

.recent-chat-btn {
    text-align: left;
    padding: 8px;
    margin: 4px 0;
    background-color: #24283b;
    border-radius: 5px;
    cursor: pointer;
}

.recent-chat-btn:hover {
    background-color: #2f354d;
}

@keyframes rotateOrbital {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes botAnimate {
    from { transform: scale(1) rotate(0deg); }
    to { transform: scale(1.1) rotate(-5deg); }
}

@keyframes slideBg {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

/* Hide Streamlit elements on landing */
[data-testid="stSidebarContent"].landing {
    display: none;
}
</style>
""", unsafe_allow_html=True)

def landing_page():
    st.markdown("""
    <div class="landing-container">
        <div class="left-section">
            <h1 class="gradient-text" style="font-size:80px;">Yamlify</h1>
            <h2>Supercharge your creativity and productivity</h2>
            <h3>Your AI-powered YAML configuration assistant. Generate, validate, and optimize your configurations with ease.</h3>
        </div>
        <div class="right-section">
            <div class="bot-container">
                <div class="bot-bg"></div>
                <img src="https://i.postimg.cc/J7gJC6bB/bot.jpg" class="bot-image"/>
                <div class="chat-box">
                    <img src="https://i.postimg.cc/T3jWmTBr/human.jpg" class="chat-avatar"/>
                    <span>Human: Let's generate some YAML!</span>
                </div>
            </div>
        </div>
        <img src="https://raw.githubusercontent.com/yourusername/yamlify/main/public/orbital.png" class="orbital"/>
        
    </div>
    """, unsafe_allow_html=True)
    
    if st.button('Get Started', key='get_started'):
        st.session_state.page = 'dashboard'
        st.rerun()

def dashboard():
    # Sidebar
    with st.sidebar:
        st.image("yamlify.jpg")
        st.title("Dashboard")
        
        # Navigation
        st.markdown("### Navigation")
        if st.button("Create a new Chat", key="new_chat"):
            clear_chat_selection()
        
        st.button("Explore Yamlify", key="explore")
        st.button("Contact", key="contact")
        
        # Recent Chats
        st.markdown("### Recent Chats")
        if 'conversation_history' in st.session_state:
            for i, (query, _) in enumerate(reversed(st.session_state.conversation_history)):
                chat_preview = query[:30] + "..." if len(query) > 30 else query
                if st.button(chat_preview, key=f"chat_{i}"):
                    handle_chat_click(len(st.session_state.conversation_history) - 1 - i)
        
        st.markdown("---")
        st.button("🌟 Upgrade to Pro", key="upgrade")

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("YAML Configuration Generator")
        
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        
        if st.session_state.selected_chat is not None:
            selected_query, selected_response = st.session_state.conversation_history[st.session_state.selected_chat]
            st.session_state.user_input = selected_query
        
        user_input = st.text_area(
            "Ask me anything...",
            value=st.session_state.user_input,
            height=100,
            placeholder="Enter your YAML configuration requirement here..."
        )
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if st.button("Generate Configuration", key="generate"):
            if user_input.strip():
                with st.spinner("Generating configuration..."):
                    session_id = "user_123"
                    response = asyncio.run(generate_config(
                        user_input,
                        "firstChat",
                        st.session_state.yaml_chain_with_memory,
                        st.session_state.fallback_chain_with_memory
                    ))
                    
                    st.session_state.conversation_history.append((user_input, response.yaml_config))
                    
                    st.markdown("### Generated YAML Configuration")
                    st.code(response.yaml_config, language='yaml')
                    
                    with st.expander("Show Details"):
                        st.markdown("#### Confidence Score")
                        st.progress(response.confidence_score)
                        
                        st.markdown("#### Sources")
                        if response.sources:
                            for source in response.sources:
                                st.markdown(f"- {source}")
                        else:
                            st.markdown("No sources available.")
        
        if st.session_state.selected_chat is not None:
            st.markdown("### Previous Response")
            st.code(st.session_state.conversation_history[st.session_state.selected_chat][1], language='yaml')

def handle_chat_click(index):
    st.session_state.selected_chat = index

def clear_chat_selection():
    st.session_state.selected_chat = None
    st.session_state.user_input = ""

async def generate_config(requirement: str, session_id: str, yaml_chain_with_memory: Any, fallback_chain_with_memory: Any) -> YAMLResponse:
    try:
        docs_list = doc_list()
        if not docs_list:
            return YAMLResponse(
                yaml_config="# No documentation available",
                confidence_score=0.0,
                sources=[],
                explanation="Failed to load documentation sources"
            )

        vectorstore = vectordb(docs_list, embeddings)
        if not vectorstore:
            return YAMLResponse(
                yaml_config="# Error: Failed to create vector database",
                confidence_score=0.0,
                sources=[],
                explanation="Internal error: Vector database creation failed"
            )

        search_results = await hybrid_search(requirement, vectorstore)
        confidence = np.mean([score for _, score in search_results]) if search_results else 0.0
        
        config = {"configurable": {"session_id": session_id}}
        
        if confidence > 0.4 and search_results:
            context = "\n".join([doc.page_content for doc, _ in search_results])
            input_message = HumanMessage(content=f"Generate YAML configuration based on: {requirement}")

            response = yaml_chain_with_memory.invoke(
                [input_message],
                config=config
            )

            return YAMLResponse(
                yaml_config=response,
                confidence_score=confidence,
                sources=[doc.metadata.get('source', '') for doc, _ in search_results],
                explanation="Configuration generated based on documentation."
            )
        
        input_message = HumanMessage(content=f"Generate YAML configuration based on: {requirement}")
        
        fallback_response = fallback_chain_with_memory.invoke(
            [input_message],
            config=config
        )
        
        return YAMLResponse(
            yaml_config=fallback_response,
            confidence_score=confidence,
            sources=[],
            explanation="Configuration generated based on general knowledge."
        )
        
    except Exception as e:
        return YAMLResponse(
            yaml_config="# Error occurred while generating configuration",
            confidence_score=0.0,
            sources=[],
            explanation=f"Error: {str(e)}"
        )

def main():
    if st.session_state.page == 'landing':
        landing_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()