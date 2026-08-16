# email_handler.py
"""
Sends transactional emails (signup verification, password reset) via SMTP.

Requires SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL
and FRONTEND_URL to be set in the environment. If SMTP isn't configured yet
(e.g. local development), the message is logged instead of sent so the auth
flow can still be exercised end-to-end without a real mailbox.
"""
import os
import smtplib
from email.message import EmailMessage

from utils.logger import get_logger

logger = get_logger(__name__, "email.log")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


def _send_email(to_email: str, subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM_EMAIL", smtp_username)

    if not smtp_host or not smtp_port or not smtp_username or not smtp_password:
        logger.warning(
            f"SMTP is not configured; logging email instead of sending. "
            f"To: {to_email} | Subject: {subject}\n{body}"
        )
        return

    message = EmailMessage()
    message["From"] = smtp_from
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(message)
        logger.info(f"Sent email '{subject}' to {to_email}")
    except Exception:
        logger.exception(f"Failed to send email '{subject}' to {to_email}")
        raise


def send_verification_email(to_email: str, token: str):
    link = f"{FRONTEND_URL}/verify-email?token={token}"
    subject = "Verify your Study Companion account"
    body = (
        "Welcome to Study Companion!\n\n"
        "Please verify your email address by clicking the link below:\n"
        f"{link}\n\n"
        "This link expires in 24 hours."
    )
    _send_email(to_email, subject, body)


def send_password_reset_email(to_email: str, token: str):
    link = f"{FRONTEND_URL}/reset-password?token={token}"
    subject = "Reset your Study Companion password"
    body = (
        "We received a request to reset your Study Companion password.\n\n"
        f"Click the link below to choose a new password:\n{link}\n\n"
        "This link expires in 1 hour. If you didn't request this, you can ignore this email."
    )
    _send_email(to_email, subject, body)
