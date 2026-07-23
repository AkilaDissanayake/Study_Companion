import logging
import os
import jwt
import datetime
import mimetypes
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Response, Request,Form,BackgroundTasks,Query,Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv

# Import custom utilities
from utils.logger import get_logger
from utils.json_handler import *
from utils.file_handler import *
from utils.vdb_handler import embed_uploaded_file,delete_file_from_vdb,delete_subject_from_vdb
from utils.database_handler import engine, Base,get_db
from utils.db_models import TokenUsage, ChatSession
from utils.response_handler import success_response, raise_api_error 
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

#Classes
class ChatRequest(BaseModel):
    """The expected JSON payload from the React frontend."""
    raw_question: str
    chat_history: Optional[str] = ""

# ==========================================
# AUTHENTICATION & JWT LOGIC
# ==========================================

async def get_current_user_from_cookie(request: Request) -> str:
    """Dependency to extract user_id securely from the HttpOnly Cookie."""
    token = request.cookies.get("session_token")
    
    if not token:
        logger.warning("Rejected request: Missing session token cookie.")
        raise_api_error(status_code=401, message="Not authenticated. Missing session token.")
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
        
    except jwt.ExpiredSignatureError:
        logger.warning("Rejected request: Session token expired.")
        raise_api_error(status_code=401, message="Session token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        logger.error("Rejected request: Invalid or tampered token detected.")
        raise_api_error(status_code=401, message="Invalid or tampered token detected.")


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
        return success_response(
            message="Login successful", 
            data={"user_id": user_id, "config": user_config}
        )
        
    except Exception as e:
        raise_api_error(status_code=401, message="Invalid Google token", error_details=e)

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
    return success_response(message="Logged out successfully")

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
        return success_response(message=f"Config {payload.filename} created successfully.")
    
    except Exception as e:
        logger.exception(f"Failed to create config {payload.filename}")
        raise_api_error(status_code=500, message=f"Failed to create config {payload.filename}", error_details=e)


@app.patch("/config/edit")
async def edit_existing_config(
    payload: ConfigPayload, 
    user_id: str = Depends(get_current_user_from_cookie)
):
    try:
        logger.debug(f"User {user_id} attempting to update config: {payload.filename}")
        update_config(payload.filename, payload.data)
        logger.info(f"Successfully updated config: {payload.filename}")
        return success_response(message=f"Config {payload.filename} updated successfully.")
        
    except Exception as e:
        logger.exception(f"Failed to update config {payload.filename}")
        raise_api_error(status_code=500, message=f"Failed to update config {payload.filename}", error_details=e)

@app.get("/config/subjects")
async def view_config(user_id: str = Depends(get_current_user_from_cookie)):
    """Endpoint to view the subjects of current user's config."""
    try:
        config_file = f"{user_id}.json"
        user_config = read_config(config_file, default_fallback={})
        logger.debug(f"User {user_id} retrieved their config.")
        subjects = list(user_config.get("subjects", []))
        return success_response(message="Subjects retrieved", data={"subjects": subjects})
    except Exception as e:
        logger.exception(f"Failed to retrieve config for user {user_id}")
        raise_api_error(status_code=500, message="Failed to retrieve config", error_details=e)

@app.get("/config/get")
async def get_full_config(user_id: str = Depends(get_current_user_from_cookie)):
    """Endpoint to return the full user configuration."""
    config_file = f"{user_id}.json"
    # Returns the JSON file content, or an empty dict if not found
    config_data = read_config(config_file, default_fallback={})
    return success_response(message="Config retrieved", data=config_data)


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

    return success_response(
        message=f"Processed {len(files)} files into {folder or 'root'}.",
        data={"saved_successfully": saved_files, "failed_to_save": failed_files}
    )


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
            print(all_files)
    return success_response(
        message="Files retrieved successfully",
        data={"total_files": len(all_files), "files": all_files}
    )


@app.get("/files/download")
async def download_user_file(
    filename: str = Query(..., description="The name of the file to download or preview"),
    subject: str = Query("root", description="The subject folder the file belongs to"),
    user_id: str = Depends(get_current_user_from_cookie)
):
    """
    Securely serves an uploaded file for viewing (inline) or downloading.
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

        raise_api_error(status_code=400, message="Security Alert: Invalid file path sequence.")

    # Check if the file actually exists on the server disk
    if not os.path.exists(normalized_path) or os.path.isdir(normalized_path):
    
        raise_api_error(status_code=404, message="The requested file could not be found.")

    logger.info(f"Serving file {filename} from subject '{subject}' to user {user_id} for preview")

    # 1. Guess the correct media type based on the file extension (e.g., 'application/pdf', 'image/jpeg')
    content_type, _ = mimetypes.guess_type(normalized_path)

    # Return the file using FileResponse
    # media_type uses the guessed type so the browser knows how to render it (PDF, images, etc.).
    # content_disposition_type="inline" tells the browser to display it in the iframe/window instead of downloading.
    # filename=filename ensures that if the user clicks "Save As" from the preview, it still has the right name.
    return FileResponse(
        path=normalized_path, 
        media_type=content_type or "application/octet-stream", 
        filename=filename,
        content_disposition_type="inline"
    )

# ==========================================
# Deletion API
# ==========================================
@app.delete("/files/delete")
async def delete_user_file(
    data: dict = Body(...),
    user_id: str = Depends(get_current_user_from_cookie)
):

    """
    Securely deletes an uploaded file.
    Prevents directory traversal attacks by reconstructing the path strictly within the user's directory.
    """
    try:
        file_name = data.get("filename")
        subject = data.get("subject", "root")
        
        # Reconstruct the base path based on whether the file is in a subject folder or root
        if subject == "root":
            file_path = os.path.join("uploads", user_id, file_name)
        else:
            file_path = os.path.join("uploads", user_id, subject, file_name)
        # Delete from upload dir
        pdf_status=delete_file(file_path)
        # Delete from vdb
        vdb_status=delete_file_from_vdb(user_id,file_name,subject)

        return success_response(message=f"File {file_name} successfully deleted.") #check
    except Exception as e:
        raise_api_error(status_code=500, message="Failed to delete file", error_details=e)

@app.delete("/files/deletesubject")
async def delete_user_subject(
    data: dict = Body(...),
    user_id: str = Depends(get_current_user_from_cookie)
):

    """
    Securely deletes an uploaded file.
    Prevents directory traversal attacks by reconstructing the path strictly within the user's directory.
    """
    try:
        subject = data.get("subject")
        
        # Reconstruct the base path based on whether the file is in a subject folder or root
        if subject!="root" : #No button for root folder
            file_path = os.path.join("uploads", user_id, subject)
            # Delete from upload dir
            pdf_status=delete_directory(file_path)
            # Delete from vdb
            vdb_status=delete_subject_from_vdb(user_id,subject)
            #After deleting subject we need to remove that subject from config file
            file_name=f"{user_id}.json"
            subjects=read_config(file_name).get("subjects")
            if subject in subjects:
                subjects.remove(subject)
            new_data={"subjects":subjects}
            
            update_config(file_name,new_data)
            
        else:
            
            file_path=os.path.join("uploads",user_id)
            pdf_status=delete_directory(file_path)
            # Delete from vdb
            vdb_status=delete_subject_from_vdb(user_id,"root")
        return success_response(message=f"Subject '{subject}' successfully deleted.")
    except Exception as e:
        raise_api_error(status_code=500, message="Failed to delete subject", error_details=e)


# ==========================================
# Chat API
# ==========================================
@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest, 
    user_id: str = Depends(get_current_user_from_cookie)
):
    """
    Main endpoint that triggers the Adaptive CRAG LangGraph state machine.
    """
    try:
        logger.info(f"Received query from user {user_id}: {request.raw_question[:50]}...")
        
        # 1. Initialize the starting state for LangGraph
        # We only need to provide the initial inputs. The graph will populate the rest.
        initial_state = {
            "user_id": user_id,
            "raw_question": request.raw_question,
            "chat_history": request.chat_history,
        }
        
        # 2. Execute the Graph
        # .invoke() runs the state machine synchronously from start to finish
        final_state = ChatBot.invoke(initial_state)
        
        # 3. Extract the outputs safely using .get()
        response_text = final_state.get("final_response", "Error: No response generated.")
        
        crag_status = final_state.get("status", "BYPASSED_CRAG") 
        confidence = final_state.get("confidence_score", 0.0)
        used_tools = final_state.get("needs_tools", False)
        subject = final_state.get("subject", "Unknown")
        detail_level = final_state.get("detail_level", "Unknown")
        
        logger.info(f"Successfully generated response for user {user_id}. Status: {crag_status}")
        
        # PACKAGE THE DATA INTO A DICTIONARY FIRST
        chat_data = {
            "response": response_text,
            "status": crag_status,
            "confidence": confidence,
            "used_tools": used_tools,
            "subject": subject,
            "detail_level": detail_level
        }
        
        
        # Return the structured JSON to the frontend
        return success_response(message="Chat generated successfully", data=chat_data)
        
    except Exception as e:
        # USE RAISE_API_ERROR HERE:
        raise_api_error(status_code=500, message="Internal Server Error during LangGraph execution.", error_details=e)


# ==========================================
# SERVER EXECUTION
# ==========================================

if __name__ == "__main__":
    logger.info("Starting up the Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)