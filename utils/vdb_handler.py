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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# Import custom utilities
from utils.logger import get_logger
from utils.file_handler import extract_pdf_text, read_text

# Configure logger
logger = get_logger(__name__, "vdb_handler.log")

# Initialize ChromaDB 
VDB_DIR = os.getenv("VDB_DIR", "vectorstore")  # Default to 'vdb' if not set
if not os.path.exists(VDB_DIR):
    os.makedirs(VDB_DIR)

chroma_client = chromadb.PersistentClient(path=VDB_DIR)


# Collections are dynamically handled per user.

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Splits text recursively by paragraphs, sentences, and words to maintain 
    semantic context while enforcing strict character limits to prevent truncation.
    """
    if not text:
        return []
        
    # 800 characters safely maps to ~200 tokens, well under Chroma's 256 limit
    logger.debug(f"Chunking text of length {len(text)} with chunk_size={chunk_size} and overlap={overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return splitter.split_text(text)

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
        #Used the default embedding model
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
def delete_file_from_vdb(user_id: str, filename: str,subject:str) -> bool:
    """
    Deletes all vector chunks associated with a specific filename 
    from a user's dedicated ChromaDB collection.
    """
    try:
        # 1. Identify the user's specific collection
        safe_user_id = str(user_id).replace("-", "_")
        collection_name = f"user_{safe_user_id}"
        
        # 2. Get the collection
        user_collection = chroma_client.get_collection(name=collection_name)
        
        # 3. Delete all chunks where the filename matches
        # This prevents 'ghost data' remaining in the vector store
        user_collection.delete(where={"$and":[{"filename":filename},{"subject":subject}]})
        
        logger.info(f"Successfully removed all vector chunks for: {filename} from {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error removing vectors for {filename}: {e}")
        return False

def delete_subject_from_vdb(user_id: str, subject: str) -> bool:
    """
    Deletes all vector chunks associated with a specific subject 
    from a user's dedicated ChromaDB collection.
    """
    try:
        safe_user_id = str(user_id).replace("-", "_")
        collection_name = f"user_{safe_user_id}"
        user_collection = chroma_client.get_collection(name=collection_name)
        
        # Filter by subject only
        user_collection.delete(where={"subject": subject})
        
        logger.info(f"Successfully removed all vectors for subject: {subject} from {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error removing vectors for subject {subject}: {e}")
        return False

def search_vdb(user_id: str, subject: str, query: str, k: int = 10) -> List[str]:
    """
    Retrieves the top-k most relevant text chunks from the user's collection.
    
    Args:
        user_id (str): The unique identifier for the user.
        subject (str): The specific subject to filter by (e.g., 'Biology').
        query (str): The rewritten user question.
        k (int): Number of initial chunks to retrieve. Set high (10) because 
                 the CRAG Grader node will aggressively filter them down.
                 
    Returns:
        List[str]: A list of raw text chunks. Returns empty list if none found.
    """
    try:
        safe_user_id = str(user_id).replace("-", "_")
        collection_name = f"user_{safe_user_id}"
        
        # 1. Safely attempt to get the user's collection
        try:
            user_collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"Collection {collection_name} does not exist. User likely hasn't uploaded documents yet.")
            return []
            
        # 2. Query ChromaDB with Metadata Filtering
        # We enforce a strict WHERE clause so the DB only searches the active subject.
        results = user_collection.query(
            query_texts=[query],
            n_results=k,
            where={"subject": subject}
        )
        
        # 3. Extract the documents
        # ChromaDB returns a list of lists for 'documents': e.g., [['chunk1', 'chunk2', ...]]
        documents = results.get("documents", [[]])[0]
        
        logger.info(f"Retrieved {len(documents)} chunks from VDB for query: '{query[:30]}...'")
        return documents
        
    except Exception as e:
        logger.error(f"Error during VDB search for user {user_id}: {e}")
        return []