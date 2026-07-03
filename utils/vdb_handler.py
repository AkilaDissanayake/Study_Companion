"""
Vector Database (ChromaDB) Handler module.

This module manages the extraction, chunking, and embedding of text documents into 
ChromaDB. It uses a "One Collection Per User" architecture to ensure strict data 
isolation and security between users. 

It handles re-uploads safely by utilizing a "Clean Slate" method, which deletes 
existing document chunks before upserting new ones to prevent ghost data.
"""
import os
import chromadb
from typing import List, Optional

# Import custom utilities
from utils.logger import get_logger
from utils.file_handler import extract_pdf_text, read_text

# Configure logger
logger = get_logger(__name__, "vdb_handler.log")

# Initialize ChromaDB 
VDB_DIR = "vectorstore"
if not os.path.exists(VDB_DIR):
    os.makedirs(VDB_DIR)

chroma_client = chromadb.PersistentClient(path=VDB_DIR)


# Collections are dynamically handled per user.

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Chops a massive document into smaller chunks.
    The overlap ensures sentences at the edge of a chunk aren't cut in half without context.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward, but step back slightly by the overlap amount
        start = end - overlap 
        
    return chunks

def embed_uploaded_file(filepath: str, user_id: str, subject: str, filename: str) -> bool:
    """
    Reads the file, extracts the text, chunks it, and embeds it into ChromaDB.
    Creates or retrieves a dedicated, isolated collection for the specific user.
    """
    try:
        logger.debug(f"Starting embedding process for: {filename} (User: {user_id}, Subject: {subject})")
        
        # Dynamically grab or create THIS specific user's collection
        # Chroma requires collection names (alphanumeric and underscores, 3-63 chars)
        safe_user_id = str(user_id).replace("-", "_")
        collection_name = f"user_{safe_user_id}" 
        
        user_collection = chroma_client.get_or_create_collection(name=collection_name)
        # If the file already exists in the collection, delete it first to avoid duplicates
        logger.info(f"Checking for existing data for {filename}...")
        user_collection.delete(where={"filename": filename})

        #  Extract Text based on file type
        raw_text = ""
        if filepath.lower().endswith(".pdf"):
            raw_text = extract_pdf_text(filepath)
        elif filepath.lower().endswith(".txt"):
            raw_text = read_text(filepath)
        else:
            logger.warning(f"Unsupported file type for embedding: {filename}")
            return False
            
        if not raw_text:
            logger.warning(f"No readable text found in {filename}")
            return False

        # Chop the text into chunks
        chunks = chunk_text(raw_text)
        logger.info(f"Generated {len(chunks)} chunks for {filename}")

        # Prepare data arrays for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            
    
            metadatas.append({ 
                "subject": subject,
                "filename": filename,
                "chunk_index": i
            })
            
            # Every chunk needs a mathematically unique ID
            ids.append(f"{subject}_{filename}_chunk_{i}")   # Unique ID for each chunk
            # Can avoid embedding duplicates if user changes doc and upload again its also handled.

        # Save to the USER'S specific Vector Database collection
        user_collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids   # Preserve the unique IDs for traceability
        )
        
        logger.info(f"Successfully embedded {filename} into {collection_name}.")
        return True

    except Exception as e:
        logger.exception(f"Failed to embed file {filename} for user {user_id}")
        return False