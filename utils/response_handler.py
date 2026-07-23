import os
from typing import Any, Dict, Optional
from fastapi import HTTPException
from utils.logger import get_logger

logger = get_logger(__name__, "api_responses.log")

# ==========================================
# PERFORMANCE FIX: Evaluate this ONCE at server startup
# ==========================================
# This evaluates to a simple True or False boolean.
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "production").lower() == "development"


def success_response(message: str, data: Optional[Dict[str, Any]] = None) -> dict:
    """Standardizes all successful API responses."""
    response = {
        "status": "success",
        "message": message
    }
    if data is not None:
        response["data"] = data
        
    return response


def raise_api_error(status_code: int, message: str, error_details: Optional[Any] = None):
    """Standardizes all error responses."""
    error_payload = {
        "status": "error",
        "message": message
    }
    
    logger.error(f"API Error [{status_code}]: {message} | Details: {error_details}")
    
    # PERFORMANCE FIX: We are now just checking a boolean variable. 
    # This takes roughly 0.0000001 seconds to execute!
    if error_details and IS_DEVELOPMENT:
        error_payload["details"] = str(error_details)
        
    raise HTTPException(status_code=status_code, detail=error_payload)