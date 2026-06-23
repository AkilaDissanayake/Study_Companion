import os
from utils.logger import get_logger
from typing import Generator, Optional

# Attempt to import PyPDF2, but don't crash if it's missing until a PDF function is called
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


# Initialize the isolated logger for this specific file
logger = get_logger(__name__, "file_handler.log")
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


def stream_text_by_chunks(filepath: str, chunk_size_mb: int = 50) -> Generator[str, None, None]:
    """
    Reads a massive file in fixed size memory blocks (default: 1 MB).
    Best for raw speed when reading, copying, or regex scanning.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Text file not found for streaming: {filepath}. No data to yield.")
        return

    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    with open(filepath, 'r', encoding='utf-8') as file:
        logger.info(f"Streaming text file in {chunk_size_mb} MB chunks: {filepath}")
        while True:
            chunk = file.read(chunk_size_bytes)
            if not chunk:
                break
            yield chunk


def stream_text_by_paragraphs(filepath: str) -> Generator[str, None, None]:
    """
    Reads a file and yields one complete paragraph at a time.
    Best for Natural Language Processing (NLP) or summarizing articles.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Text file not found for paragraph streaming: {filepath}. No data to yield.")
        return

    with open(filepath, 'r', encoding='utf-8') as file:
        paragraph = []
        
        for line in file:
            if line.strip() == "":
                if paragraph:
                    yield "".join(paragraph).strip()
                    paragraph = []
            else:
                paragraph.append(line)
                
        if paragraph:
            yield "".join(paragraph).strip()


# ==========================================
# PDF FILE UTILITIES
# ==========================================

def extract_pdf_text(filepath: str) -> Optional[str]:
    """
    Opens a compiled PDF and attempts to extract all readable text.
    Requires PyPDF2.
    """
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is not installed")

    if not os.path.exists(filepath):
        logger.warning(f"PDF not found: {filepath}")
        return None

    extracted_text = []

    try:
        # 'rb' stands for Read Binary. PDFs are not standard text files!
        with open(filepath, 'rb') as file:
            logger.info(f"Extracting text from PDF: {filepath}")
            reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
                    
        return "\n\n".join(extracted_text)
        
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return None