import numpy as np
from typing import List, Dict, Any
from rag import hybrid_search  # Importing hybrid_search from rag module
from result_format import YAMLResponse  # Importing YAMLResponse from result_format
from define_params import SessionManager, create_chain # Importing creating_chain from define_params
from data_loading import doc_list, vectordb  # Importing doc_list and vectordb from data_loading
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
import os

google_api_key = "AIzaSyAw4gUH6WEbGZaBg6KSLoYfU_djx-mEjxY"
import asyncio
from logging import getLogger



# Initialize logging
logger = getLogger(__name__)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

async def generate_config(
    requirement: str,
    session_id: str,
    yaml_chain_with_memory: Any,
    fallback_chain_with_memory: Any
) -> YAMLResponse:
    
    try:
        # Load documents
        logger.info("Loading documents...")
        docs_list = doc_list()
        if not docs_list:
            logger.warning("No documents were loaded successfully")
            return YAMLResponse(
                yaml_config="# No documentation available",
                confidence_score=0.0,
                sources=[],
                explanation="Failed to load documentation sources"
            )

        # Create vector database
        logger.info("Creating vector database...")
        vectorstore = vectordb(docs_list, embeddings)
        if not vectorstore:
            logger.error("Failed to create vector database")
            return YAMLResponse(
                yaml_config="# Error: Failed to create vector database",
                confidence_score=0.0,
                sources=[],
                explanation="Internal error: Vector database creation failed"
            )

        # Perform hybrid search
        logger.info("Performing hybrid search...")
        search_results = await hybrid_search(requirement, vectorstore)
        confidence = np.mean([score for _, score in search_results]) if search_results else 0.0
        
        # Configuration for chain invocation
        config = {"configurable": {"session_id": session_id}}
        
        # If we have relevant documentation with good confidence
        if confidence > 0.4 and search_results:
            logger.info(f"Found relevant documentation with confidence {confidence}")
            
            # Prepare context and input
            context = "\n".join([doc.page_content for doc, _ in search_results])
            input_message = HumanMessage(content=f"Generate YAML configuration based on: {requirement}")

            # Invoke the primary chain
            response = yaml_chain_with_memory.invoke(
                [input_message],
                config=config
            )

            return YAMLResponse(
                yaml_config=response,
                confidence_score=confidence,
                sources=[doc.metadata.get('source', '') for doc, _ in search_results],
                explanation="Configuration generated based on Spheron ICL documentation and examples."
            )
        
        # Fallback to general ICL knowledge
        logger.info("Using fallback chain due to low confidence or no results")
        input_message = HumanMessage(content=f"Generate YAML configuration based on: {requirement}")
        
        fallback_response = fallback_chain_with_memory.invoke(
            [input_message],
            config=config
        )
        
        return YAMLResponse(
            yaml_config=fallback_response,
            confidence_score=confidence,
            sources=[],
            explanation="Configuration generated based on general ICL knowledge. Please verify against Spheron documentation."
        )
        
    except Exception as e:
        logger.error(f"Error generating configuration: {str(e)}", exc_info=True)
        return YAMLResponse(
            yaml_config="# Error occurred while generating configuration",
            confidence_score=0.0,
            sources=[],
            explanation=f"Error: {str(e)}"
        )

async def main():
    try:
        # Initialize session management
        session_manager = SessionManager()
        
        # Create chains
        yaml_chain_with_memory, fallback_chain_with_memory = create_chain(session_manager)
        
        # Example usage
        requirement = "Create a Node.js server with 4 CPU"
        session_id = "user_123"  # In practice, this should be generated uniquely per user
        
        response = await generate_config(
            requirement=requirement,
            session_id=session_id,
            yaml_chain_with_memory=yaml_chain_with_memory,
            fallback_chain_with_memory=fallback_chain_with_memory
        )
        
        print("\nGenerated YAML Configuration:")
        print(response.yaml_config)
        print("\nConfidence Score:", response.confidence_score)
        print("\nSources:", response.sources)
        print("\nExplanation:", response.explanation)
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    
    asyncio.run(main())
