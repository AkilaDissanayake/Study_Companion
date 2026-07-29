"""
Web Search Utility Module.
Handles external web queries, result parsing, and relevance grading.
"""
import math
from typing import Tuple, Any
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from utils.logger import get_logger

logger = get_logger(__name__, "web_search.log")

def search_and_grade_web(query: str, grader_model: Any, max_results: int = 5) -> Tuple[str, bool]:
    """
    Executes an academic web search, extracts URLs and snippets, 
    grades them for relevance using the local Cross-Encoder, 
    and formats the verified results with their source links.
    
    Returns:
        Tuple[str, bool]: The formatted context string, and a boolean indicating success.
    """
    logger.info(f"Performing external web search for: {query}")
    wrapper = DuckDuckGoSearchAPIWrapper()
    
    # Domain Restriction: Force reliable academic/informational sources
    academic_query = f"{query} site:edu OR site:wikipedia.org OR site:scholar.google.com"
    
    try:
        # Fetch structured results (snippet, title, link)
        raw_web_results = wrapper.results(academic_query, max_results=max_results)
        
        if not raw_web_results:
            raise ValueError("No results found via DuckDuckGo")
            
        refined_snippets = []
        logger.debug("Raw web structured results received.")
        
        # Iterate over the structured results
        for result in raw_web_results:
            snippet = result.get("snippet", "")
            link = result.get("link", "Unknown Web Source")
            
            if len(snippet.strip()) < 15:
                continue  # Skip empty or tiny fragments
                
            # Grade the snippet using the provided local Cross-Encoder
            raw_score = float(grader_model.predict([query, snippet.strip()]))
            try:
                confidence = (1 / (1 + math.exp(-raw_score))) * 100
            except OverflowError:
                confidence = 0.0 if raw_score < 0 else 100.0
                
            # Keep web snippets that are at least somewhat relevant (e.g., > 40%)
            if confidence >= 40.0:
                refined_snippets.append(f"[Source: {link}]\n{snippet.strip()}")
        
        # If the grader threw everything away, the search was a failure
        if not refined_snippets:
            raise ValueError("Web search returned irrelevant noise.")
            
        # Recompose the surviving, high-quality facts with their URLs
        formatted_web_data = "[Verified External Web Knowledge]:\n\n" + "\n\n".join(refined_snippets)
        return formatted_web_data, True
        
    except Exception as e:
        logger.warning(f"Web search failed or was rejected by grader: {e}")
        return "[External Web Search Failed]", False