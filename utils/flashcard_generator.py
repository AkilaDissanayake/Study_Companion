# flashcard_generator.py
"""
AI flashcard deck generation from an uploaded document's extracted text.

A single-purpose, non-LangGraph utility (mirrors utils/image_captioner.py's
shape) — flashcard generation doesn't need the quiz system's drafter/critic
fact-verification loop (models/quiz_generator.py), just one JSON-mode call.
"""
import json

from langchain_openai import ChatOpenAI
from utils.prompts import FLASHCARD_GENERATOR_PROMPT
from utils.token_manager import log_token_usage
from utils.logger import get_logger

logger = get_logger(__name__, "flashcard_generator.log")

flashcard_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind(response_format={"type": "json_object"})

# Bounds cost/latency for very large documents — a truncation of the source
# text, not the resulting card count. Full-document coverage via a
# map-reduce over chunks would be a genuinely bigger feature than this one.
MAX_SOURCE_CHARS = 12000


def generate_flashcards_from_text(text: str, user_id: str) -> list:
    """Returns a list of {"front": ..., "back": ...} dicts, or [] on
    failure — same fail-soft contract as image_captioner.caption_image."""
    if not text:
        return []

    try:
        chain = FLASHCARD_GENERATOR_PROMPT | flashcard_llm
        response = chain.invoke({"source_text": text[:MAX_SOURCE_CHARS]})

        try:
            token_usage = response.response_metadata.get("token_usage", {})
            log_token_usage(
                user_id=user_id,
                model_name="gpt-4o-mini-flashcards",
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
            )
        except Exception:
            logger.warning("Failed to log token usage for flashcard generation")

        parsed = json.loads(response.content)
        cards = parsed.get("cards", [])
        return [c for c in cards if isinstance(c, dict) and c.get("front") and c.get("back")]

    except Exception as e:
        logger.warning(f"Failed to generate flashcards: {e}")
        return []
