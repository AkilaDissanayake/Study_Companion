import os
import shutil
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
    