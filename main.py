import logging
import os
import jwt
import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Response, Request,Form,BackgroundTasks,Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, List
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv

# Import custom utilities
from utils.logger import get_logger
from utils.json_handler import *
from utils.file_handler import *
from utils.vdb_handler import embed_uploaded_file
from utils.database_handler import engine, Base,get_db
from utils.db_models import TokenUsage, ChatSession 
from models.chatbot import ChatBot
# Load environment variables (.env)
load_dotenv()

# Initialize the isolated logger for this file
logger = get_logger(__name__, "main.log")



# Initialize FastAPI
app = FastAPI(title="Study Companion API")

# Configure CORS so the Vite React frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Must match  exact frontend URL
    allow_credentials=True, # Required to send the secure HttpOnly cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI application successfully started.")

# Secure Keys from .env
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-me")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# ==========================================
# AUTHENTICATION & JWT LOGIC
# ==========================================

async def get_current_user_from_cookie(request: Request) -> str:
    """Dependency to extract user_id securely from the HttpOnly Cookie."""
    token = request.cookies.get("session_token")
    
    if not token:
        logger.warning("Rejected request: Missing session token cookie.")
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
        
    except jwt.ExpiredSignatureError:
        logger.warning("Rejected request: Session token expired.")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        logger.error("Rejected request: Invalid or tampered token detected.")
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/login/google")
async def login_with_google(token: str, response: Response):
    """Verifies Google token, retrieves config, and sets a 7-day secure cookie."""
    try:
        # 1. Verify Google Identity
        id_info = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        user_id = id_info.get("sub")
        
        # 2. Get/Create User Config
        config_file = f"{user_id}.json"
        user_config = read_config(config_file, default_fallback={"theme": "light"})
        
        # 3. Issue Custom JWT valid for 7 days
        jwt_payload = {
            "sub": user_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }
        custom_jwt = jwt.encode(jwt_payload, SECRET_KEY, algorithm="HS256")
        
        # 4. Set Secure HttpOnly Cookie
        response.set_cookie(
            key="session_token", 
            value=custom_jwt, 
            httponly=True,  
            secure=True,    
            samesite="lax"
        )
        
        logger.info(f"User {user_id} successfully logged in.")
        return {
            "user_id": user_id,
            "config": user_config
        }
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Google token")

@app.post("/login/logout")
async def logout_user(response: Response):
    """
    Clears the HttpOnly cookie to securely log the user out.
    """
    response.delete_cookie(
        "session_token", 
        httponly=True, 
        samesite="lax"
    )
    logger.info("User successfully logged out and cookie cleared.")
    return {"message": "Logged out successfully"}

@app.get("/auth/check")
async def check_auth(user_id: str = Depends(get_current_user_from_cookie)):
    """Validates the session cookie on page refresh."""
    return {"authenticated": True, "user_id": user_id}

# ==========================================
# PYDANTIC MODELS (For JSON Body Validation)
# ==========================================

class ConfigPayload(BaseModel):
    filename: str
    data: Dict[str, Any]


# ==========================================
# CONFIGURATION APIs
# ==========================================

@app.post("/config/create")
async def create_config(
    payload: ConfigPayload, 
    user_id: str = Depends(get_current_user_from_cookie)
):
    try:
        logger.debug(f"User {user_id} attempting to create config: {payload.filename}")
        write_config(payload.filename, payload.data)
        logger.info(f"Successfully created config: {payload.filename}")
        return {"message": f"Config {payload.filename} created successfully."}
    
    except Exception as e:
        logger.exception(f"Failed to create config {payload.filename}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/config/edit")
async def edit_existing_config(
    payload: ConfigPayload, 
    user_id: str = Depends(get_current_user_from_cookie)
):
    try:
        logger.debug(f"User {user_id} attempting to update config: {payload.filename}")
        update_config(payload.filename, payload.data)
        logger.info(f"Successfully updated config: {payload.filename}")
        return {"message": f"Config {payload.filename} updated successfully."}
        
    except Exception as e:
        logger.exception(f"Failed to update config {payload.filename}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/subjects")
async def view_config(user_id: str = Depends(get_current_user_from_cookie)):
    """Endpoint to view the subjects of current user's config."""
    try:
        config_file = f"{user_id}.json"
        user_config = read_config(config_file, default_fallback={})
        logger.debug(f"User {user_id} retrieved their config.")
        subjects = list(user_config.get("subjects", []))
        return {"subjects": subjects}
    except Exception as e:
        logger.exception(f"Failed to retrieve config for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/get")
async def get_full_config(user_id: str = Depends(get_current_user_from_cookie)):
    """Endpoint to return the full user configuration."""
    config_file = f"{user_id}.json"
    # Returns the JSON file content, or an empty dict if not found
    return read_config(config_file, default_fallback={})


# ==========================================
# SECURE MULTIPLE FILE UPLOAD API
# ==========================================

@app.post("/files/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    folder: str = Form(None), # Captures the subject name from the frontend
    user_id: str = Depends(get_current_user_from_cookie) 
):
    """
    Receives multiple uploaded files, saves them into a strictly 
    segregated user/subject directory, and automatically updates 
    the user's saved subjects in their config file.
    """
    # 1. Dynamically build the path based on whether a subject was provided
    if folder:
        upload_dir = f"uploads/{user_id}/{folder}"
        
        # --- NEW: Automatically update the User's Config File ---
        try:
            config_filename = f"{user_id}.json"
            # Read existing config to get the current list of subjects
            user_config = read_config(config_filename, default_fallback={"subjects": []})
            current_subjects = list(user_config.get("subjects", []))
            
            # If this is a brand new subject, add it and save the config!
            if folder not in current_subjects:
                current_subjects.append(folder)
                update_config(config_filename, {"subjects": current_subjects})
                logger.info(f"Added new subject '{folder}' to {user_id}'s config.")
                
        except Exception as e:
            logger.error(f"Failed to update subjects for {user_id}: {str(e)}")
            
    else:
        # Fallback to the root user directory if no subject is selected
        upload_dir = f"uploads/{user_id}"
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    saved_files = []
    failed_files = []

    for file in files:
        try:
            logger.debug(f"Processing file upload for {user_id} in {folder or 'root'}: {file.filename}")
            filepath = os.path.join(upload_dir, file.filename)
            contents = await file.read()
            
            # Use appropriate handler based on file extension
            if file.filename.endswith(".txt"):
                text_content = contents.decode('utf-8')
                write_text_safe(filepath, text_content) 
            else:
                with open(filepath, "wb") as f:
                    f.write(contents)
            
            saved_files.append(filepath)
            logger.info(f"Successfully saved {file.filename} for {user_id}")
            # Schedule the embedding process in the background and send the response immediately to the frontend.
            background_tasks.add_task(
            embed_uploaded_file, 
            filepath=filepath, 
            user_id=user_id, 
            subject=folder or "root", 
            filename=file.filename
        )

        except Exception as e:
            logger.exception(f"Failed to process file {file.filename} for {user_id}")
            failed_files.append(file.filename)

    return {
        "message": f"Processed {len(files)} files for {user_id} into {folder or 'root'}. They are now processing",
        "saved_successfully": saved_files,
        "failed_to_save": failed_files
    }


# ==========================================
# API's to list and send files
# ==========================================

@app.get("/files/names")
async def get_user_file_names(user_id: str = Depends(get_current_user_from_cookie)):
    """
    Retrieves ALL file names uploaded by a specific user.
    Recursively scans the user's root upload folder and all subject subfolders.
    """
    # Define the absolute root of this user's storage
    base_dir = f"uploads/{user_id}"
    
    # If the directory doesn't exist, they haven't uploaded anything yet
    if not os.path.exists(base_dir):
        return {
            "user_id": user_id, 
            "total_files": 0, 
            "files": []
        }

    all_files = []

    # os.walk looks at the current folder, then dives into every subfolder inside it
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Ignore hidden system files
            if file.startswith("."):
                continue
                
            # Determine which folder this file is sitting in (for UI categorization)
            relative_path = os.path.relpath(root, base_dir)
            folder_name = "root" if relative_path == "." else relative_path
            
            all_files.append({
                "filename": file,
                "subject": folder_name
            })

    return {
        "user_id": user_id,
        "total_files": len(all_files),
        "files": all_files
    }


@app.get("/files/download")
async def download_user_file(
    filename: str = Query(..., description="The name of the file to download"),
    subject: str = Query("root", description="The subject folder the file belongs to"),
    user_id: str = Depends(get_current_user_from_cookie)
):
    """
    Securely serves an uploaded file for viewing or downloading.
    Prevents directory traversal attacks by reconstructing the path strictly within the user's directory.
    """
    # Reconstruct the base path based on whether the file is in a subject folder or root
    if subject == "root":
        file_path = os.path.join("uploads", user_id, filename)
    else:
        file_path = os.path.join("uploads", user_id, subject, filename)

    # Security Check: Prevent directory traversal attacks (e.g., filename="../../../etc/passwd")
    # This ensures the resolved path strictly stays inside the 'uploads' folder
    normalized_path = os.path.normpath(file_path)
    if not normalized_path.startswith("uploads"):
        logger.warning(f"Security Alert: User {user_id} attempted directory traversal with path: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid file path sequence.")

    # Check if the file actually exists on the server disk
    if not os.path.exists(normalized_path) or os.path.isdir(normalized_path):
        logger.warning(f"File not found on disk: {normalized_path} for user {user_id}")
        raise HTTPException(status_code=404, detail="The requested file could not be found.")

    logger.info(f"Serving file {filename} from subject '{subject}' to user {user_id}")

    # Return the file using FileResponse
    # media_type="application/octet-stream" forces a browser download dialog.
    # filename=filename ensures the browser saves it with its original human-readable name.
    return FileResponse(
        path=normalized_path, 
        media_type="application/octet-stream", 
        filename=filename
    )
# ==========================================
# SERVER EXECUTION
# ==========================================

if __name__ == "__main__":
    logger.info("Starting up the Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)