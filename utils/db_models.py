# db_models.py
"""
Data Blueprint and Schema Definitions Module.

Contains the SQLAlchemy Object Relational Mapping  models representing 
the strict structured relational token ledger and the flexible semi structured 
PostgreSQL JSONB conversational storage tables.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB, ARRAY  

from utils.database_handler import Base

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