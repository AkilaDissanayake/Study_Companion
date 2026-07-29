"""
External tools for the Study Companion app.
Includes LRU caching for expensive operations and async execution wrappers.
"""

import io
import base64
import os
import asyncio
from functools import lru_cache
import numexpr as ne
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from langchain_core.tools import tool
from utils.vdb_handler import chroma_client
from utils.logger import get_logger

logger = get_logger(__name__, "tools.log")


# =========================================================
# LRU CACHED HELPERS
# =========================================================

@lru_cache(maxsize=100)
def _cached_dalle_generate(prompt: str, quality: str) -> str:
    """
    Internal synchronous helper cached with LRU.
    Stores up to 100 recent image generation URLs in memory.
    If the exact same prompt and quality are requested again, it returns 
    the cached URL instantly without invoking the OpenAI API.
    """
    logger.info(f"[LRU Cache Miss] Generating new image via OpenAI for prompt: '{prompt[:30]}...'")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality=quality,
        n=1,
    )
    return response.data[0].url


# =========================================================
# ASYNC LANGCHAIN TOOLS
# =========================================================

# --- 1. Precision Calculator ---
@tool
async def precision_calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the precise numerical result.
    Use this for arithmetic, algebra, and trigonometry instead of guessing.
    Example expressions: '3 * (4 + 5)', 'sin(3.14)', '2**10'
    """
    try:
        # numexpr safely evaluates math strings in C without arbitrary code execution risks
        result = ne.evaluate(expression)
        return f"The exact result is: {result}"
    except Exception as e:
        return f"Error evaluating math expression: {e}. Ensure it is formatted correctly."


# --- 2. Graph Generator ---
@tool
async def generate_math_graph(equation: str, x_min: int = -10, x_max: int = 10) -> str:
    """
    Generates a visual graph for a mathematical equation (e.g., 'x**2', 'np.sin(x)').
    Returns a Markdown base64 image link.
    """
    def _plot():
        x = np.linspace(x_min, x_max, 400)
        safe_dict = {
            "x": x, "np": np, 
            "sin": np.sin, "cos": np.cos, "tan": np.tan, 
            "exp": np.exp, "sqrt": np.sqrt, "log": np.log
        }
        
        y = eval(equation, {"__builtins__": None}, safe_dict)
        
        # --- THREAD-SAFE MATPLOTLIB ---
        # Create a specific figure and axis for THIS thread only
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, color="#3498db", linewidth=2)
        ax.set_title(f"Graph of y = {equation}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight') # save 'fig', not 'plt'
        buf.seek(0)
        plt.close(fig) # close this specific figure
        # ------------------------------
        
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"![Math Graph](data:image/png;base64,{base64_img})"

    try:
        # Run CPU-bound Matplotlib rendering off the main thread
        return await asyncio.to_thread(_plot)
    except Exception as e:
        return f"Error generating graph for '{equation}': {e}. Use standard math notation."


# --- 3. AI Image Generator ---
@tool
async def generate_image(prompt: str, high_quality: bool = False) -> str:
    """
    Generates an educational image or diagram based on a text prompt using DALL-E 3.
    Use this when the user asks for a picture, diagram, or visual representation of a concept.
    
    Parameters:
    - prompt: The detailed description of the image to generate.
    - high_quality: Set this to True ONLY if the user explicitly asks for a 'high quality', 
      'high resolution', '4k', or 'highly detailed' image. Otherwise, leave it False.
    """
    try:
        quality_setting = "hd" if high_quality else "standard"
        
        # Offload API execution to thread pool while checking LRU cache
        image_url = await asyncio.to_thread(_cached_dalle_generate, prompt, quality_setting)
        return f"![Generated Image]({image_url})"
    except Exception as e:
        return f"Error generating image: {e}"


# --- 4. List Documents Tool ---
@tool
async def list_documents(user_id: str) -> str:
    """
    Lists all the unique filenames of the documents the user has uploaded to their workspace.
    """
    def _list():
        safe_user_id = str(user_id).replace("-", "_")
        collection_name = f"user_{safe_user_id}"
        
        try:
            col = chroma_client.get_collection(name=collection_name)
            data = col.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            
            filenames = set()
            for meta in metadatas:
                if meta and "filename" in meta:
                    filenames.add(meta["filename"])
                    
            if not filenames:
                return "The user currently has no documents uploaded in their workspace."
                
            formatted_list = "\n".join([f"- {f}" for f in filenames])
            return f"Here are the user's uploaded documents:\n{formatted_list}"
            
        except Exception:
            return "No document collection found. The user has not uploaded any files yet."

    # Run disk I/O off the main thread
    return await asyncio.to_thread(_list)


# Export tools array for LangGraph binding
tools = [precision_calculator, generate_math_graph, generate_image, list_documents]