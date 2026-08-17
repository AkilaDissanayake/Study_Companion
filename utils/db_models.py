# db_models.py
"""
Data Blueprint and Schema Definitions Module.

Contains the SQLAlchemy Object Relational Mapping  models representing 
the strict structured relational token ledger and the flexible semi structured 
PostgreSQL JSONB conversational storage tables.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DateTime,ForeignKey,JSON, Float, Text
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


class UserStats(Base):
    """
    Non-derivable per-user state for the motivational/gamification layer.
    Everything else (streak count, quiz totals, avg score, etc.) is computed
    on read from TokenUsage/ChatSession/QuizRecord — this table only stores
    what genuinely can't be derived: which milestones the user has already
    been shown, so a celebration only fires once.

    One row per user, created lazily (get-or-create) on first stats read —
    this is a brand-new table, not a column added to the existing `users`
    table, since this app has no migration framework and ALTERs on existing
    tables are silently never applied by Base.metadata.create_all().
    """
    __tablename__ = "user_stats"

    user_id = Column(String, primary_key=True)
    last_seen_streak_milestone = Column(Integer, default=0, nullable=False)
    last_seen_badge_ids = Column(ARRAY(String), default=list, nullable=False)
    timezone = Column(String, nullable=True)  # opportunistic from client; defaults to UTC until set

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Flashcard(Base):
    """
    A single spaced-repetition card, reviewed via the classic SM-2 algorithm
    (ease_factor/interval_days/repetitions — see utils/srs_handler.py).
    Fed from two sources into one shared review queue: an AI-generated deck
    parsed from an uploaded file (source="upload", source_ref=filename), or
    automatically when a quiz question is answered wrong
    (source="quiz", source_ref=quiz_id) — see the grading hook in main.py's
    /quizzes/{quiz_id}/grade.
    """
    __tablename__ = "flashcards"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    subject = Column(String, nullable=True)
    front = Column(Text, nullable=False)
    back = Column(Text, nullable=False)
    source = Column(String, nullable=False)       # "upload" | "quiz"
    source_ref = Column(String, nullable=True)     # filename (upload) or quiz_id (quiz)

    ease_factor = Column(Float, default=2.5, nullable=False)
    interval_days = Column(Float, default=0, nullable=False)
    repetitions = Column(Integer, default=0, nullable=False)
    next_review_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())