# utils/tools.py
"""
Agent Tools Definition.

Contains all external functions and capabilities that the LangGraph agent can invoke.
Keeping these isolated prevents the main graph orchestration script from becoming bloated.
"""

from langchain_core.tools import tool

@tool
def DrawGraph(equation: str) -> str:
    """Renders visual graphs for mathematical equations."""
    # Future: Connect to a graphing library or frontend config generator
    return f"[Simulated Graph Data Rendered for: {equation}]"

@tool
def GenerateImage(prompt: str) -> str:
    """Generates an image based on a descriptive prompt."""
    # Future: Connect to DALL-E or Stable Diffusion API
    return f"[Simulated Image Generated for: {prompt}]"

# Export a single list of all available tools so the main graph can easily bind them
tools = [ DrawGraph, GenerateImage]