# database_handler.py
"""
Database Connection Module.

This module initializes the core SQLAlchemy engine and configures the session 
 used to connect the FastAPI application to the production PostgreSQL instance.
It also exposes the operational database session dependency.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environmental variables from the .env file
load_dotenv()

# Retrieve the native PostgreSQL connection string from environment context
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the core SQLAlchemy engine bound to  PostgreSQL instance.
# Set echo=True if need to debug raw SQL operations in  terminal.
engine = create_engine(DATABASE_URL, echo=False)

# Establish a session factory for generating thread local database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class that mappings will inherit from to form SQL tables
Base = declarative_base()
#Create tables if they don't exist yet
print("Checking database tables...")
Base.metadata.create_all(bind=engine)

def get_db():
    """
    FastAPI dependency that provides an isolated database transaction session.
    Automatically closes and cleans up the connection once the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()