"""
Image Captioning module.

Sends images embedded in uploaded PDFs to a vision-capable LLM to produce a
highly detailed text description, so diagrams/charts/figures that PyPDF2's
text-only extraction would otherwise silently drop become searchable content
in the vector store (see utils/file_handler.py's extract_pdf_text).
"""
import base64

from langchain_openai import ChatOpenAI
from utils.prompts import IMAGE_CAPTION_PROMPT
from utils.token_manager import log_token_usage
from utils.logger import get_logger

logger = get_logger(__name__, "image_captioner.log")

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
}


def _guess_mime_type(image_name: str) -> str:
    ext = image_name.rsplit(".", 1)[-1].lower() if "." in image_name else ""
    return _EXT_TO_MIME.get(ext, "image/png")


def caption_image(image_bytes: bytes, image_name: str, user_id: str) -> str | None:
    """Sends one embedded PDF image to a vision LLM and returns a highly
    detailed text description, or None if captioning fails — the caller
    (extract_pdf_text) skips the image and keeps going rather than failing
    the whole document over one bad image.
    """
    try:
        mime_type = _guess_mime_type(image_name)
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"

        chain = IMAGE_CAPTION_PROMPT | vision_llm
        response = chain.invoke({"image_data_uri": data_uri})

        try:
            token_usage = response.response_metadata.get("token_usage", {})
            log_token_usage(
                user_id=user_id,
                model_name="gpt-4o-vision-captioner",
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
            )
        except Exception:
            logger.warning(f"Failed to log token usage for image caption of {image_name}")

        return response.content

    except Exception as e:
        logger.warning(f"Failed to caption image {image_name}: {e}")
        return None
