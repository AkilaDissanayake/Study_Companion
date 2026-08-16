# security.py
"""
Password hashing/verification and single-use token generation for the
email/password auth flow (signup, login, email verification, password reset).
"""
import secrets

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def generate_token() -> str:
    """Generates a cryptographically strong, URL-safe single-use token
    (used for email verification links and password reset links)."""
    return secrets.token_urlsafe(32)
