import logging
import os
import jwt
import datetime
import mimetypes
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Response, Request,Form,BackgroundTasks,Query,Body
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, List, Optional
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

# Import custom utilities
from utils.logger import get_logger
from utils.json_handler import *
from utils.file_handler import *
from utils.vdb_handler import embed_uploaded_file,delete_file_from_vdb,delete_subject_from_vdb
from utils.database_handler import engine, Base,get_db
from utils.db_models import TokenUsage, ChatSession, QuizRecord, User
from utils.response_handler import success_response, raise_api_error
from utils.database_handler import engine, Base
from utils import db_models
from utils.security import hash_password, verify_password, generate_token
from utils.email_handler import send_verification_email, send_password_reset_email
from utils.pricing_handler import get_pricing_tiers
from models.chatbot import ChatBot
from models.quiz_generator import QuizGeneratorAgent
# Load environment variables (.env)
load_dotenv()



# Initialize the isolated logger for this file
logger = get_logger(__name__, "main.log")

Image_dir=os.getenv("IMAGE_DIR","images")
logger.info(f"Ensuring image directory exists at: {Image_dir}")
os.makedirs(Image_dir,exist_ok=True)


if os.getenv("RESET_DB", "False").lower() == "true":
    #Remove tables and create new tables
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Executing table creation in main...")
# Now that the models are registered, create the tables
logger.debug(f"Registered tables before creation: {Base.metadata.tables.keys()}")
Base.metadata.create_all(bind=engine)
# ==========================================
# WEEKLY BACKGROUND UPDATER
# ==========================================
def run_model_updater():
    """
    Runs the Hugging Face update script in a completely isolated subprocess.
    This ensures that the os.environ overrides in the script DO NOT leak 
    into the main FastAPI server's memory!
    """
    logger.info("[Scheduler] Starting weekly Hugging Face model update check...")
    try:
        # sys.executable ensures it uses your exact virtual environment's Python
        subprocess.run([sys.executable, "utils/cache_models.py"], check=True)
        logger.info("[Scheduler] Model update check completed successfully.")
    except Exception as e:
        logger.error(f"[Scheduler] Model update check failed: {e}")

# ==========================================
# FASTAPI LIFESPAN MANAGER
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- ON STARTUP ---
    scheduler = BackgroundScheduler()
    
    # 1. Fetch from .env, with safe fallbacks just in case the .env is missing
    update_day = os.getenv("MODEL_UPDATE_DAY", "sun")
    
    # 2. Wrap the hour in int() because os.getenv always returns a string ("3" -> 3)
    try:
        update_hour = int(os.getenv("MODEL_UPDATE_HOUR", 3))
    except ValueError:
        logger.warning("Invalid MODEL_UPDATE_HOUR in .env. Defaulting to 3 AM.")
        update_hour = 3

    # 3. Inject the dynamic variables into the cron scheduler
    scheduler.add_job(run_model_updater, 'cron', day_of_week=update_day, hour=update_hour, minute=0)
    scheduler.start()
    
    logger.info(f"Background task scheduler started. Updates scheduled for {update_day.capitalize()} at {update_hour}:00.")
    
    yield # The FastAPI server is actively running here
    
    # --- ON SHUTDOWN ---
    scheduler.shutdown()
    logger.info("Background task scheduler cleanly shut down.")


# Initialize FastAPI
app = FastAPI(title="Study Companion API",lifespan=lifespan)
#Mount the directory to the FastAPI app for serving images
app.mount("/generated_images", StaticFiles(directory="images"), name="images")
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
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is not set in the environment. Refusing to start with an insecure default."
    )
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

#Classes
class ChatRequest(BaseModel):
    raw_question: str
    session_id: Optional[str] = None  # None means "create a new chat"

class QuizSubmission(BaseModel):
    answers: Dict[str, List[str]]

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class VerifyEmailRequest(BaseModel):
    token: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


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


def issue_session_cookie(response: Response, user_id: str):
    """Issues the shared 7-day session cookie used by both the Google and
    email/password login flows."""
    jwt_payload = {
        "sub": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    custom_jwt = jwt.encode(jwt_payload, SECRET_KEY, algorithm="HS256")
    response.set_cookie(
        key="session_token",
        value=custom_jwt,
        httponly=True,
        secure=True,
        samesite="lax"
    )


@app.post("/login/google")
async def login_with_google(token: str, response: Response, db: Session = Depends(get_db)):
    """Verifies Google token, gets/creates the local User record, retrieves
    config, and sets a 7-day secure cookie."""
    try:
        # 1. Verify Google Identity
        id_info = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        user_id = id_info.get("sub")
        email = id_info.get("email")
        name = id_info.get("name") or (email.split("@")[0] if email else "User")

        # 2. Get or create the local User record for this Google account
        user = db.query(User).filter(User.google_id == user_id).first()
        if not user:
            # A different account (local email/password signup) may already
            # own this email — do not silently merge identities.
            existing = db.query(User).filter(User.email == email).first()
            if existing:
                raise_api_error(
                    status_code=409,
                    message="This email is already registered with a password. Please sign in with your password instead.",
                )
            user = User(
                id=user_id,
                email=email,
                name=name,
                google_id=user_id,
                password_hash=None,
                is_email_verified=True,  # Google has already verified this email
            )
            db.add(user)
            db.commit()
            logger.info(f"Created new local User record for Google account {user_id}.")

        # 3. Get/Create User Config
        config_file = f"{user_id}.json"
        user_config = read_config(config_file, default_fallback={"theme": "light"})

        # 4. Issue the shared session cookie
        issue_session_cookie(response, user_id)

        logger.info(f"User {user_id} successfully logged in via Google.")
        return success_response(
            message="Login successful",
            data={"user_id": user_id, "email": user.email, "name": user.name, "config": user_config}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise_api_error(status_code=401, message="Invalid Google token", error_details=e)


@app.post("/auth/signup")
async def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    """Creates a new email/password account and emails a verification link.
    Does not log the user in — they must verify their email first."""
    try:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            method = "Google sign-in" if existing.password_hash is None else "your password"
            raise_api_error(status_code=409, message=f"This email is already registered. Please sign in with {method}.")

        token = generate_token()
        user = User(
            id=str(uuid.uuid4()),
            email=payload.email,
            name=payload.name,
            password_hash=hash_password(payload.password),
            google_id=None,
            is_email_verified=False,
            email_verification_token=token,
            email_verification_expires=datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        )
        db.add(user)
        db.commit()

        send_verification_email(user.email, token)
        logger.info(f"New local signup for {user.email}; verification email sent.")
        return success_response(message="Account created. Please check your email to verify your account.")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise_api_error(status_code=500, message="Failed to create account", error_details=e)


@app.post("/auth/verify-email")
async def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    """Marks an account's email as verified using the token from the emailed link."""
    user = db.query(User).filter(User.email_verification_token == payload.token).first()
    if not user or not user.email_verification_expires or user.email_verification_expires < datetime.datetime.utcnow():
        raise_api_error(status_code=400, message="This verification link is invalid or has expired.")

    user.is_email_verified = True
    user.email_verification_token = None
    user.email_verification_expires = None
    db.commit()

    logger.info(f"Email verified for user {user.id}.")
    return success_response(message="Email verified successfully. You can now log in.")


@app.post("/login")
async def login_with_password(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """Authenticates an email/password account and sets the shared session cookie."""
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not user.password_hash:
        raise_api_error(status_code=401, message="Invalid email or password.")

    if not verify_password(payload.password, user.password_hash):
        raise_api_error(status_code=401, message="Invalid email or password.")

    if not user.is_email_verified:
        raise_api_error(status_code=403, message="Please verify your email before logging in.")

    issue_session_cookie(response, user.id)

    config_file = f"{user.id}.json"
    user_config = read_config(config_file, default_fallback={"theme": "light"})

    logger.info(f"User {user.id} successfully logged in via password.")
    return success_response(
        message="Login successful",
        data={"user_id": user.id, "email": user.email, "name": user.name, "config": user_config}
    )


@app.post("/auth/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Always returns the same generic response to avoid leaking whether an
    email is registered. Only sends an email for local (password-based) accounts."""
    user = db.query(User).filter(User.email == payload.email).first()
    if user and user.password_hash:
        token = generate_token()
        user.password_reset_token = token
        user.password_reset_expires = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        db.commit()
        send_password_reset_email(user.email, token)
        logger.info(f"Password reset requested for user {user.id}.")

    return success_response(message="If an account with that email exists, a password reset link has been sent.")


@app.post("/auth/reset-password")
async def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Sets a new password using the token from the emailed reset link."""
    user = db.query(User).filter(User.password_reset_token == payload.token).first()
    if not user or not user.password_reset_expires or user.password_reset_expires < datetime.datetime.utcnow():
        raise_api_error(status_code=400, message="This password reset link is invalid or has expired.")

    user.password_hash = hash_password(payload.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    db.commit()

    logger.info(f"Password reset completed for user {user.id}.")
    return success_response(message="Password reset successfully. You can now log in with your new password.")


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
async def check_auth(user_id: str = Depends(get_current_user_from_cookie), db: Session = Depends(get_db)):
    """Validates the session cookie on page refresh."""
    user = db.query(User).filter(User.id == user_id).first()
    return {
        "authenticated": True,
        "user_id": user_id,
        "email": user.email if user else None,
        "name": user.name if user else None,
    }

# ==========================================
# PYDANTIC MODELS (For JSON Body Validation)
# ==========================================

class ConfigPayload(BaseModel):
    filename: str
    data: Dict[str, Any]


# ==========================================
# PRICING API (public, no auth — hand-edited pricing.yaml)
# ==========================================

@app.get("/pricing")
async def get_pricing():
    """Serves the hand-edited pricing config for the landing page. No auth,
    no DB — plan/trial tracking and quota enforcement are separate, later work."""
    try:
        config = get_pricing_tiers()
        return success_response(message="Pricing retrieved", data=config)
    except Exception as e:
        logger.exception("Failed to load pricing config")
        raise_api_error(status_code=500, message="Failed to load pricing", error_details=e)


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
    Securely deletes a subject folder, its vector DB entries, and configuration.
    Prevents directory traversal attacks by reconstructing the path strictly within the user's directory.
    """
    try:
        subject = data.get("subject")
        
        #  Validation: Ensure subject was actually provided
        if not subject:
            return raise_api_error(status_code=400, message="Subject name is required.")
            
        #  Security: Strictly prevent directory traversal attacks (e.g., subject="../another_user")
        if "/" in subject or "\\" in subject or ".." in subject:
             return raise_api_error(status_code=400, message="Invalid subject name.")

        #  Main Logic
        if subject != "root":
            # Delete from upload dir
            file_path = os.path.join("uploads", user_id, subject)
            pdf_status = delete_directory(file_path)
            
            # Delete from vdb
            vdb_status = delete_subject_from_vdb(user_id, subject)
            
            # Read config safely, defaulting to an empty list if 'subjects' doesn't exist
            file_name = f"{user_id}.json"
            config_data = read_config(file_name) or {}
            subjects = config_data.get("subjects", [])
            
            # Safely remove and update
            if subject in subjects:
                subjects.remove(subject)
                # Ensure we don't overwrite other unrelated data in the config
                config_data["subjects"] = subjects
                update_config(file_name, config_data)
                
            return success_response(message=f"Subject '{subject}' successfully deleted.")
            
        else:
            #  Prevent deletion of the 'root' folder
            return raise_api_error(
                status_code=403, 
                message="You do not have permission to delete the 'root' folder."
            )
            
    except Exception as e:
        #   Handle any error occcuring during deletion
        
        return raise_api_error(status_code=500, message="Failed to delete subject", error_details=e)

# ==========================================
# Chat API
# ==========================================

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest, 
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """
    Main endpoint that triggers the Adaptive CRAG LangGraph state machine.
    """
    try:
        # --- SESSION MANAGEMENT ---
        if request.session_id:
            # Fetch existing chat
            chat_session = db.query(ChatSession).filter(
                ChatSession.id == request.session_id, 
                ChatSession.user_id == user_id
            ).first()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="Chat session not found")
        else:
            # Create a brand new chat
            chat_session = ChatSession(user_id=user_id, chat_state=[])
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            
        # Extract history for LangGraph
        current_history = chat_session.chat_state 

        # ---  LANGGRAPH EXECUTION ---
        logger.debug(f"Invoking LangGraph for user {user_id} with question: {request.raw_question} and session {chat_session.id}")
        initial_state = {
            "user_id": user_id,
            "raw_question": request.raw_question,
            "chat_history": current_history, # Pass DB history to your agent
        }
        
        final_state =await ChatBot.ainvoke(initial_state)
        response_text = final_state.get("final_response", "Error: No response generated.")
        
        # ---  SAVE NEW MESSAGES TO JSONB ---
        # Append the new user and AI messages to the history list
        chat_session.chat_state.append({"role": "user", "content": request.raw_question})
        chat_session.chat_state.append({"role": "ai", "content": response_text})
        
        # SQLAlchemy requires this flag when mutating a JSONB dictionary/list in place
        logger.debug(f"Updating chat_state for session {chat_session.id} with new messages.")
        flag_modified(chat_session, "chat_state") 
        db.commit()

        # ---  RETURN RESPONSE ---
        chat_data = {
            "session_id": chat_session.id, # Frontend MUST save this to use on the next prompt
            "response": response_text,
            "status": final_state.get("status", "BYPASSED_CRAG"),
            "used_tools": final_state.get("needs_tools", False)
        }
        
        return success_response(message="Chat generated", data=chat_data)

    except Exception as e:
        db.rollback()
        raise_api_error(status_code=500, message="Internal Server Error", error_details=e)

@app.get("/chats")
async def get_user_chats(
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Fetches all chat sessions for the sidebar UI."""
    logger.info(f"Fetching all chat sessions for user_id: {user_id}")
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id)\
                 .order_by(ChatSession.updated_at.desc()).all()

    logger.debug(f"Found {len(sessions)} chat sessions for user_id: {user_id}")
    # Return just the ID, the first message as a "title", and the timestamp
    chat_list = []
    for s in sessions:
        title = "New Chat"
        if len(s.chat_state) > 0:
            # Grab the first 30 characters of the very first user prompt
            title = s.chat_state[0].get("content", "New Chat")[:30] + "..."
            
        chat_list.append({
            "session_id": s.id,
            "title": title,
            # Convert datetime to a JSON friendly string
            "updated_at": s.updated_at.isoformat() if s.updated_at else None 
        })
        
    return success_response(message="Chats fetched successfully", data=chat_list)


@app.get("/chats/{session_id}")
async def get_single_chat(
    session_id: str,
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Fetches the full message history for a specific chat."""
    chat = db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        ChatSession.user_id == user_id
    ).first()
    logger.info(f"Fetching chat history for session_id: {session_id} (user_id: {user_id})")
    if not chat:
        raise_api_error(
            status_code=404, 
            message="Chat not found", 
            error_details=f"The chat session '{session_id}' does not exist or you do not have permission to access it."
        )
        
    # --- Format the JSONB data for React ---
    formatted_history = []
    for index, msg in enumerate(chat.chat_state):
        # Convert Langchain's 'ai' role to your frontend's 'bot' role
        role = "bot" if msg.get("role") in ["ai", "assistant"] else "user"
        
        formatted_history.append({
            "id": f"history-{index}",  # Generate a unique ID for React's mapping
            "role": role,
            "text": msg.get("content", "")  # Map 'content' to 'text'
        })
        
    return success_response(message="Chat fetched successfully", data={
        "session_id": chat.id,
        "chat_state": formatted_history
    })


@app.delete("/chats/{session_id}")
async def delete_chat_session(
    session_id: str,
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Deletes a specific chat session for the authenticated user."""
    logger.info(f"Deleting chat session_id: {session_id} for user_id: {user_id}")
    
    chat = db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        ChatSession.user_id == user_id
    ).first()
    
    if not chat:
        logger.warning(f"Chat not found or unauthorized delete attempt: session_id={session_id}, user_id={user_id}")
        raise_api_error(status_code=404, message="Chat not found", error_details=f"The chat session '{session_id}' does not exist or you do not have permission to access it.")  #hANDLE

        
    db.delete(chat)
    db.commit()
    
    logger.info(f"Successfully deleted chat session_id: {session_id}")
    return success_response(message="Chat deleted successfully")

# ==========================================
# Quiz API
# ==========================================

@app.post("/chats/{session_id}/quiz")
async def generate_quiz_from_chat(
    session_id: str,
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Generates a zero-hallucination quiz using LangGraph, saves answers, and sends sanitized questions."""
    logger.info(f"Triggering LangGraph Quiz Agent for session_id: {session_id}")
    
    try:
        # 1. Fetch the chat
        chat = db.query(ChatSession).filter(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        ).first()
        
        if not chat or not chat.chat_state:
            return raise_api_error(
                status_code=404, 
                message="Chat not found", 
                error_details=f"The chat session '{session_id}' does not exist or is empty."
            )
            
        history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat.chat_state])
        
        # 2. INVOKE THE LANGGRAPH AGENT
        initial_state = {
            "chat_history": history_text,
            "retry_count": 0,
            "critique": ""
        }
        
        # The agent will loop internally until the Critic node outputs "PASS"
        final_state = QuizGeneratorAgent.invoke(initial_state)
        
        # Extract the verified quiz from the final state
        full_quiz_data = final_state["draft_quiz"]
        
        # 3. Save the full data (with answers) securely to the database
        quiz_id = str(uuid.uuid4())
        new_quiz = QuizRecord(
            id=quiz_id,
            session_id=session_id,
            user_id=user_id,
            full_quiz_data=full_quiz_data
        )
        db.add(new_quiz)
        db.commit()
        
        # 4. Strip answers before sending to the React frontend
        sanitized_questions = []
        for q in full_quiz_data.get("questions", []):
            # Fallback check for older quizzes that used the "answer" string
            correct_list = q.get("correct_answers", [q.get("answer")] if q.get("answer") else [])
            
            sanitized_questions.append({
                "question": sanitize_text(q["question"]),
                "options": [sanitize_text(opt) for opt in q["options"]],
                "is_multiple_choice": len(correct_list) > 1 # True if more than 1 answer
            })

        # FIX: ADDED MISSING RETURN STATEMENT HERE!
        return success_response(message="Quiz generated successfully!", data={
            "quiz_id": quiz_id,
            "title": full_quiz_data.get("title", "Generated Quiz"),
            "questions": sanitized_questions
        })
        
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        return raise_api_error(status_code=500, message="Internal Server Error", error_details=str(e))



@app.post("/quizzes/{quiz_id}/grade")
async def grade_quiz(
    quiz_id: str,
    submission: QuizSubmission,
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    try:
        quiz_record = db.query(QuizRecord).filter(
            QuizRecord.id == quiz_id, QuizRecord.user_id == user_id
        ).first()

        if not quiz_record:
            return raise_api_error(status_code=404, message="Quiz not found", error_details="No matching quiz record.")

        full_data = quiz_record.full_quiz_data
        total_score = 0.0
        results = {}

        for idx, q in enumerate(full_data.get("questions", [])):
            idx_str = str(idx)
            
            # Convert arrays to sets for easy math comparisons
            user_ans = set(submission.answers.get(idx_str, []))
            correct_ans = set(q.get("correct_answers", [q.get("answer")] if q.get("answer") else []))
            
            # 1. Calculate hits and misses
            correct_hits = user_ans.intersection(correct_ans)
            incorrect_hits = user_ans.difference(correct_ans)
            
            # 2. Calculate partial credit (max 1 point per question)
            if len(correct_ans) > 0:
                # E.g., 2 correct answers total. User picks 1 right, 0 wrong -> 0.5 points
                # User picks 1 right, 1 wrong -> 0.0 points
                points = (len(correct_hits) - len(incorrect_hits)) / len(correct_ans)
                q_score = max(0.0, points) # Prevent negative scores on a question
            else:
                q_score = 0.0
                
            total_score += q_score
            
            # 3. Check if it was perfectly answered for the UI styling
            is_perfect = (user_ans == correct_ans)
            
            results[idx_str] = {
                "user_answers": [sanitize_text(a) for a in user_ans],
                "correct_answers": [sanitize_text(a) for a in correct_ans],
                "is_correct": is_perfect,
                "points_awarded": round(q_score, 2), # Send partial points to frontend
                "explanation": sanitize_text(q["explanation"])
            }

        return success_response(message="Quiz graded!", data={
            "score": round(total_score, 2), # Round to 2 decimal places (e.g., 2.5)
            "total": len(full_data.get("questions", [])),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Quiz grading failed: {e}")
        return raise_api_error(status_code=500, message="Internal Server Error", error_details=str(e))


        
@app.get("/quizzes")
async def get_all_quizzes(
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Fetches a lightweight list of all past quizzes for the sidebar/list view."""
    try:
        quizzes = db.query(QuizRecord).filter(QuizRecord.user_id == user_id).order_by(QuizRecord.created_at.desc()).all()
        
        quiz_list = [
            {
                "id": q.id, 
                "title": q.full_quiz_data.get("title", "Untitled Quiz"),
                "created_at": q.created_at
            } 
            for q in quizzes
        ]
        return success_response(message="Quizzes fetched", data=quiz_list)
    except Exception as e:
        return raise_api_error(status_code=500, message="Failed to fetch quizzes", error_details=str(e))

@app.get("/quizzes/{quiz_id}")
async def get_single_quiz(
    quiz_id: str,
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Fetches a specific quiz and STRIPS THE ANSWERS before sending to frontend."""
    try:
        quiz = db.query(QuizRecord).filter(
            QuizRecord.id == quiz_id, QuizRecord.user_id == user_id
        ).first()
        
        if not quiz:
            return raise_api_error(status_code=404, message="Quiz not found", error_details="Quiz does not exist.")
            
        # Securely sanitize the questions
        sanitized_questions = []
        
        # FIX: CHANGED full_quiz_data TO quiz.full_quiz_data
        for q in quiz.full_quiz_data.get("questions", []):
            correct_list = q.get("correct_answers", [q.get("answer")] if q.get("answer") else [])
            
            sanitized_questions.append({
                "question": sanitize_text(q["question"]),
                "options": [sanitize_text(opt) for opt in q["options"]],
                "is_multiple_choice": len(correct_list) > 1
            })

        # ADDED MISSING RETURN STATEMENT HERE
        return success_response(message="Quiz loaded", data={
            "quiz_id": quiz.id,
            "title": quiz.full_quiz_data.get("title", "Untitled Quiz"),
            "questions": sanitized_questions
        })

    except Exception as e:
        return raise_api_error(status_code=500, message="Failed to load quiz", error_details=str(e))


@app.delete("/quizzes/{quiz_id}")
async def delete_quiz(
    quiz_id: str,
    user_id: str = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    """Deletes a specific quiz for the authenticated user."""
    logger.info(f"Deleting quiz_id: {quiz_id} for user_id: {user_id}")

    quiz = db.query(QuizRecord).filter(
        QuizRecord.id == quiz_id, QuizRecord.user_id == user_id
    ).first()

    if not quiz:
        logger.warning(f"Quiz not found or unauthorized delete attempt: quiz_id={quiz_id}, user_id={user_id}")
        raise_api_error(status_code=404, message="Quiz not found", error_details=f"The quiz '{quiz_id}' does not exist or you do not have permission to access it.")

    db.delete(quiz)
    db.commit()

    logger.info(f"Successfully deleted quiz_id: {quiz_id}")
    return success_response(message="Quiz deleted successfully")


# ==========================================
# SERVER EXECUTION
# ==========================================

if __name__ == "__main__":
    logger.info("Starting up the Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)