# db_models.py
"""
Data Blueprint and Schema Definitions Module.

Contains the SQLAlchemy Object Relational Mapping  models representing 
the strict structured relational token ledger and the flexible semi structured 
PostgreSQL JSONB conversational storage tables.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DateTime,ForeignKey,JSON
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.sql import func

from utils.database_handler import Base


class User(Base):
    """
    Represents an account holder, created either via Google Sign-In or via
    email/password signup. Both auth methods share this single table:
    `password_hash` is NULL for Google-only accounts, `google_id` is NULL
    for local (email/password) accounts.

    `id` is the Google `sub` claim for Google-created accounts (preserving
    the identifier every other table already keys `user_id` on) or a
    generated uuid4 for local signups.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=True)
    google_id = Column(String, unique=True, index=True, nullable=True)

    is_email_verified = Column(Boolean, default=False, nullable=False)
    email_verification_token = Column(String, nullable=True)
    email_verification_expires = Column(DateTime, nullable=True)
    password_reset_token = Column(String, nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TokenUsage(Base):
    """
    Represents an immutable, append-only transaction log for token consumption.
    This behaves like a financial ledger to accurately track model usage.
    """
    __tablename__ = "token_usage"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatSession(Base):
    """
    Represents a dynamic chat session tracking historical chat sequences.
    Utilizes PostgreSQL's native JSONB format to safely store variable agent states.
    """
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    
    # THE FIX: Now accepts a list of strings! e.g., ["Physics", "Math"]
    # We rename it to 'subjects' (plural) to reflect it holds multiple tags.
    subjects = Column(ARRAY(String), default=list, index=True) 
    
    chat_state = Column(JSONB, default=list, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class QuizRecord(Base):
    __tablename__ = "quizzes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id",ondelete="CASCADE"))
    user_id = Column(String)
    full_quiz_data = Column(JSON)  # Stores the LLM output (with answers!)
    created_at = Column(DateTime(timezone=True), server_default=func.now())