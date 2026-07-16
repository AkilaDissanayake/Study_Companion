# token_manager.py
"""
Token Metrics Management Utilities.

Provides internal backend operations responsible for handling resource accounting data, 
ensuring database write tasks are decoupled securely from primary application threads.
"""
from utils.database_handler import SessionLocal #have to manage manually since this is not a FastAPI endpoint context
from utils.db_models import TokenUsage
from utils.logger import get_logger

logger = get_logger(__name__, "token_usage.log")


def log_token_usage(user_id: str, model_name: str, prompt_tokens: int, completion_tokens: int):
    """
    Creates an isolated database session to commit an atomic token log entry.
    Wrapped in try-except blocks to prevent database bottlenecks or transaction
    failures from crashing or slowing down active user chat streams.
    """
    # Open a dedicated database session separate from FastAPI endpoint threads
    db = SessionLocal()
    try:
        # Pre-compute totals before hitting the engine layer
        total = prompt_tokens + completion_tokens
        
        # Instantiate the database record row
        usage_record = TokenUsage(
            user_id=user_id,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total
        )
        
        # Add to the session unit of work and commit to the actual PostgreSQL table
        db.add(usage_record)
        db.commit()
    except Exception as e:
        # Rollback the session state immediately if database locks or exceptions happen
        db.rollback()
        # Fall back to console/app logging so engineering teams can inspect failures
        logger.error(f"Failed to log token metrics to DB: {str(e)}")
    finally:
        # Always terminate the session connection back into the Pool
        db.close()