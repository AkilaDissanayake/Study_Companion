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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI
from langchain_core.tools import tool
import uuid
from utils.vdb_handler import chroma_client
from utils.logger import get_logger

logger = get_logger(__name__, "tools.log")

IMAGE_DIR = os.getenv("IMAGE_DIR", "generated_images")
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
async def generate_math_graph(equation: str, user_id: str = "guest", x_min: float = -10, x_max: float = 10) -> str:
    """
    Generates a visual graph for a mathematical equation (e.g., 'x**2', 'sin(x)').
    Returns a standard URL link to the generated image.
    """
    def _plot():
        try:
            # 1. Sanitize common LLM syntax errors before evaluation
            # Convert JavaScript/Latex style math to Python/NumPy style
            safe_eq = equation.replace("^", "**") # Fix exponents
            safe_eq = safe_eq.replace("np.", "") # Strip np. if LLM adds it, we define sin natively
            
            x = np.linspace(x_min, x_max, 400)
            
            # 2. Evaluate using numexpr for safety and speed over arrays
            # We define local variables that numexpr can access
            y = ne.evaluate(safe_eq, local_dict={"x": x})
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y, color="#3498db", linewidth=2)
            ax.set_title(f"Graph of y = {safe_eq}")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(0, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            fig.tight_layout()
            
            # 3. Create a specific subdirectory for this user
            user_folder = os.path.join(IMAGE_DIR, str(user_id))
            os.makedirs(user_folder, exist_ok=True)
            
            # 4. Save the file inside the user's secure folder
            filename = f"graph_{uuid.uuid4().hex}.png"
            filepath = os.path.join(user_folder, filename)
            
            fig.savefig(filepath, format='png')
            plt.close(fig) 
            
            # 5. Construct the dynamic URL path
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            return f"![Math Graph]({backend_url}/generated_images/{user_id}/{filename})"
            
        except Exception as inner_e:
            # Log the specific error so you can see WHY it crashed in your console
            logger.error(f"Graphing failed inside _plot thread: {inner_e}")
            raise inner_e # Re-raise to be caught by the outer try-except

    try:
        return await asyncio.to_thread(_plot)
    except Exception as e:
        logger.error(f"Outer tool execution failed: {e}")
        return f"Error generating graph for '{equation}': {e}"
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