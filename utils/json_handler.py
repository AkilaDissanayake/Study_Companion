"""Helper functions for safely reading and writing JSON config files.
Includes:
- read_config: Safely reads a JSON config file, returning a default fallback if the file doesn't exist or is malformed.
- write_config: Safely fully overwrites a JSON file using Atomic Writes to prevent data corruption.
- update_config: Merges new settings into the existing configuration, leaving unspecified settings completely untouched."""

import json
import os
from typing import Any, Dict
from utils.logger import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__, "json_handler.log")
load_dotenv()  # Load environment variables from .env file
CONFIG_DIR = os.getenv("CONFIG_DIR", "configs")  # Default to 'configs' if not set
def read_config(filename: str, default_fallback: Dict = None) -> Dict:
    """Safely reads a JSON config file."""
    if default_fallback is None:
        default_fallback = {}
        
    filepath = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(filepath):
        logger.warning(f"Config file not found: {filepath}. Returning default fallback.")
        return default_fallback

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}. Returning default fallback.")
        return default_fallback


def write_config(filename: str, data: Dict) -> None:
    """Safely fully overwrites a JSON file using Atomic Writes."""
    if not os.path.exists(CONFIG_DIR):
        logger.info(f"Config directory not found. Creating directory: {CONFIG_DIR}")
        os.makedirs(CONFIG_DIR)
        
    filepath = os.path.join(CONFIG_DIR, filename)
    temp_filepath = filepath + ".tmp"

    try:
        with open(temp_filepath, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, indent=4)
            temp_file.flush()
            os.fsync(temp_file.fileno())
    except Exception as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise e

    os.replace(temp_filepath, filepath)


def update_config(filename: str, new_data: Dict) -> None:
    """
    Merges new settings into the existing configuration.
    Leaves unspecified settings completely untouched.
    """
    # Grab the current state of the file (or {} if it doesn't exist yet)
    current_state = read_config(filename)
    
    # Merge the new data into the current state.
    # Python's .update() method overwrites existing keys and adds new ones, 
    # but leaves unmentioned keys completely alone.
    current_state.update(new_data)
    
    # Safely save the newly merged dictionary using  writer
    write_config(filename, current_state)