"""
API Response Handler Utility.

This module standardizes the structure of all outgoing HTTP responses for the FastAPI application.
It ensures that the frontend receives a consistent JSON payload for both successful operations
and errors, making frontend parsing predictable and reducing boilerplate code.

It also handles environment aware error reporting:
- In 'development' mode, raw exceptions are passed to the frontend for easier debugging.
- In 'production' mode, raw exceptions are hidden to prevent information leakage, while
  still logging the full trace securely on the backend.
"""
import os
from typing import Any, Dict, Optional
from fastapi import HTTPException
from utils.logger import get_logger

logger = get_logger(__name__, "api_responses.log")


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
    
    
    if error_details and IS_DEVELOPMENT:
        error_payload["details"] = str(error_details)
        
    raise HTTPException(status_code=status_code, detail=error_payload)