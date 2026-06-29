import logging
import os
import jwt
import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Response, Request,Form
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

# Load environment variables (.env)
load_dotenv()

# Initialize the isolated logger for this file
logger = get_logger(__name__, "main.log")

# Initialize FastAPI
app = FastAPI(title="Study Companion API")

# Configure CORS so the Vite React frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Must match your exact frontend URL
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

# ==========================================
# SECURE MULTIPLE FILE UPLOAD API
# ==========================================

@app.post("/files/upload")
async def upload_documents(
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

        except Exception as e:
            logger.exception(f"Failed to process file {file.filename} for {user_id}")
            failed_files.append(file.filename)

    return {
        "message": f"Processed {len(files)} files for {user_id} into {folder or 'root'}.",
        "saved_successfully": saved_files,
        "failed_to_save": failed_files
    }
# ==========================================
# SERVER EXECUTION
# ==========================================

if __name__ == "__main__":
    logger.info("Starting up the Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)