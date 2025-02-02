# from langchain_community.chain import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Tuple
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

google_api_key = "AIzaSyA3KNgJIorcrjTPBYycEpoGgqPcUDKu9Us"
class SessionManager:
    def __init__(self):
        self.store: Dict[str, BaseChatMessageHistory] = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create a session history for the given session ID."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

def create_chain(session_manager: SessionManager) -> Tuple[RunnableWithMessageHistory, RunnableWithMessageHistory]:
    """Create the main and fallback chains with session management."""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.3,
        timeout=60
    )
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    # Define prompts using ChatPromptTemplate
    yaml_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in writing YAML configurations in (ICL) format supported by Spheron Network. "
                "Ensure that your YAML output follows key-value pairs with correct unit notation."
                "You are also expert in answer the query based on past result and inputs "),
        ("human", "Context from documentation:\nICL YAML configuration standards\n\n"
                "Requirement:\n{user_input}\n\n"
                "Generate a YAML configuration using the following format:\n"
                "profile_name:\n  cpu: <integer>\n  memory: \"<integer>Gi\"\n  storage: \"<integer>Gi\"\n"
                "Ensure correct unit representation (Gi for memory/storage).")
    ])
    
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in writing YAML configurations in (ICL) format supported by Spheron Network."
         "You are also expert in answer the query based on past result and inputs "),
        ("human", "Without specific context but based on general ICL knowledge, generate a YAML script for:\n\n{user_input}\n\n"
                 "Please include comments explaining each section.")
    ])
    
    # Initialize parser
    parser = StrOutputParser()
    
    # Create base chains
    yaml_chain = yaml_prompt | llm | parser
    fallback_chain = fallback_prompt | llm | parser

    # Create memory-enabled chains using the session manager
    yaml_chain_with_memory = RunnableWithMessageHistory(
        yaml_chain,
        session_manager.get_session_history
    )
    
    fallback_chain_with_memory = RunnableWithMessageHistory(
        fallback_chain,
        session_manager.get_session_history
    )

    return yaml_chain_with_memory, fallback_chain_with_memory
