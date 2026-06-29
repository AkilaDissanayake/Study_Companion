"""
Create a logger that logs to both a file and the console. 
The log file will be automatically stored in a custom 'logs' directory, 
and it will rotate when it reaches a specified size. The logger will also 
format the log messages to include the timestamp, log level, logger name, 
line number, and message.

To use:
Import the function and call it with the logger_name (__name__) and the 
log_filename as the required arguments. Optionally, you can specify the 
file size and backup count for log rotation.

ex- logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.exception("This is an exception message")

Levels:
- DEBUG: Detailed information, typically of interest only when diagnosing problems.
- INFO: Confirmation that things are working as expected.
- WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. 'disk space low'). The software is still working as expected.
- ERROR: Due to a more serious problem, the software has not been able to perform some function.
- CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
- EXCEPTION: Logs a message with level ERROR on this logger. The arguments are interpreted as for debug(), except that any passed exc_info is not ignored. This is a convenience method for logging an ERROR with exception information.
"""

import logging
import logging.handlers
import os
import sys
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Define your custom master directory for all logs
CUSTOM_LOG_DIR = os.getenv("LOG_DIR", "logs")  # Default to 'logs' if not set

def get_logger(
    logger_name: str,
    log_filename: str, 
    file_size: int = 5,
    backup_count: int = 3
) -> logging.Logger:
    """
    Creates a dedicated logger that forces its output into a custom logs directory.
    Only requires the internal logger name and the filename.
    Can change the file size and backup count if desired.
    """
    # Convert the file size from MB to Bytes (e.g., 5 MB cap before rotating)
    max_file_size = file_size * 1024 * 1024
    
    # 1. Grab a SPECIFIC logger by name (The Internal ID)
    logger = logging.getLogger(logger_name)
    
    # Prevent duplicate handlers if this function is accidentally called twice
    if logger.hasHandlers():
        return logger

    # Set baseline to DEBUG so all individual calls work natively
    logger.setLevel(logging.DEBUG)

    # Create a detailed formatter with Date, Time, Logger Name, and Line Numbers
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Force the log file into the custom 'logs' directory
    full_log_path = os.path.join(CUSTOM_LOG_DIR, log_filename)

    # Check if the directory exists, create it if it doesn't
    log_dir = os.path.dirname(full_log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Append mode ('a') resumes where it left off without overwriting
    file_handler = logging.handlers.RotatingFileHandler(
        filename=full_log_path,
        mode='a', 
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    
    # Keep console output and format it cleanly
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Attach handlers strictly to THIS specific logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # CRITICAL: Stop this logger from sending messages up the chain to the root.
    # This maintains 100% isolation for this specific file.
    logger.propagate = False

    return logger