import os
import re
import shutil
from utils.logger import get_logger
from typing import Generator, Optional

# Attempt to import PyPDF2, but don't crash if it's missing until a PDF function is called
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from utils.image_captioner import caption_image

# Skips tiny decorative icons/bullets/logos — a raw-byte-size heuristic
# rather than pixel dimensions, so this doesn't need a Pillow dependency
# just to check image size.
MIN_IMAGE_BYTES = 3000
# Cost/latency safety valve: this codebase has no rate-limit/retry infra to
# lean on otherwise, so a pathological PDF (hundreds of embedded images)
# can't turn into hundreds of sequential vision-LLM calls.
MAX_IMAGES_PER_DOC = 20


# Initialize the isolated logger for this specific file
logger = get_logger(__name__, "file_handler.log")

# Matches C0 control bytes (NUL, STX, ETX, etc.) excluding \t \n \r.
# PDFs that use non-standard/symbolic font encodings (common in math or
# scanned textbooks) sometimes make PyPDF2 emit raw control bytes instead of
# the intended glyph — left unstripped, these silently propagate into
# embeddings, chat history, and quiz content, and render as invisible/garbled
# characters on the frontend.
_CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Strips non-printable control-byte characters from extracted/stored text."""
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub('', text)

# ==========================================
# TEXT FILE UTILITIES
# ==========================================

def read_text(filepath: str, fallback: str = "") -> str:
    """
    Safely reads an entire text file. 
    Returns the fallback string if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Text file not found: {filepath}. Returning fallback.")
        return fallback

    try:
        # utf-8 prevents Windows/Mac character encoding crashes
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        return fallback


def write_text_safe(filepath: str, content: str, append: bool = False) -> None:
    """
    Writes or appends to a text file. Automatically creates missing directories.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        logger.info(f"Directory not found for {filepath}. Creating directory: {directory}")
        os.makedirs(directory)

    # 'a' appends to the end, 'w' overwrites from scratch
    mode = 'a' if append else 'w'
    
    with open(filepath, mode, encoding='utf-8') as file:
        file.write(content)

# ==========================================
# PDF FILE UTILITIES
# ==========================================

def extract_pdf_text(filepath: str, user_id: Optional[str] = None) -> Optional[str]:
    """
    Opens a compiled PDF and attempts to extract all readable text, plus a
    highly detailed AI-generated description of every embedded image (above
    a minimum size, to skip decorative icons/logos) spliced directly after
    the text of the page it appears on — so diagrams/charts/figures become
    searchable content instead of silently dropped. `user_id` attributes the
    vision-LLM token usage for image captions; captioning is skipped
    entirely if it isn't provided. Requires PyPDF2.
    """
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is not installed")

    if not os.path.exists(filepath):
        logger.warning(f"PDF not found: {filepath}")
        return None

    extracted_text = []
    images_captioned = 0

    try:
        # 'rb' stands for Read Binary. PDFs are not standard text files!
        with open(filepath, 'rb') as file:
            logger.info(f"Extracting text from PDF: {filepath}")
            reader = PyPDF2.PdfReader(file)

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_parts = []

                text = page.extract_text()
                if text:
                    page_parts.append(text)

                # Image captioning is best-effort and page-scoped: a failure
                # here degrades to text-only for this page rather than
                # aborting extraction for the whole document.
                if user_id:
                    try:
                        for image_file in page.images:
                            if images_captioned >= MAX_IMAGES_PER_DOC:
                                break
                            if len(image_file.data) < MIN_IMAGE_BYTES:
                                continue

                            caption = caption_image(image_file.data, image_file.name, user_id)
                            images_captioned += 1
                            if caption:
                                page_parts.append(f"[Image description: {caption}]")
                    except Exception as e:
                        logger.warning(f"Failed to process images on page {page_num} of {filepath}: {e}")

                if page_parts:
                    extracted_text.append("\n\n".join(page_parts))

        return sanitize_text("\n\n".join(extracted_text))
        
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return None

def delete_file(file_path: str) -> bool:
    """
    Safely deletes a file from the filesystem.
    Returns True if successfully deleted or if the file didn't exist,
    returns False if an error occurred during deletion.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File successfully deleted: {file_path}")
            return True
        else:
            logger.warning(f"Attempted to delete non-existent file: {file_path}")
            # Returning True because the end goal (file not being there) is achieved
            return True
            
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False

def delete_directory(directory_path: str) -> bool:
    """
    Safely deletes a directory and all its contents.
    """
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            logger.info(f"Directory successfully deleted: {directory_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting directory {directory_path}: {e}")
        return False
    