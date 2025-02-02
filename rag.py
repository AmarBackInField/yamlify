from typing import List, Dict, Any
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import asyncio
from logging import getLogger

async def hybrid_search(query: str, vectorstore: Chroma, k: int = 4) -> List[tuple[Document, float]]:
    """Perform hybrid search combining semantic and keyword matching."""
    try:
        yaml_keywords = ["yaml", "icl", "configuration", "deploy", "service", "resources"]
        enhanced_query = f"{query} {' '.join(yaml_keywords)}"
        
        semantic_results = await vectorstore.asimilarity_search_with_relevance_scores(
            enhanced_query, 
            k=k
        )
        
        example_results = await vectorstore.asimilarity_search_with_relevance_scores(
            "yaml example configuration template",
            k=2
        )
        
        all_results = semantic_results + example_results
        unique_results = {str(doc.page_content): (doc, score) 
                         for doc, score in all_results}
        
        return sorted(unique_results.values(), 
                     key=lambda x: x[1], 
                     reverse=True)[:k]
    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        return []