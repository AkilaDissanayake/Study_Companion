# pricing_handler.py
"""
Loads the hand-edited pricing config (pricing.yaml) that backs the public
GET /pricing endpoint. No caching — the file is tiny and rarely changes,
so re-reading it on every call lets an admin edit prices without a restart.
"""
import os

import yaml

from utils.logger import get_logger

logger = get_logger(__name__, "pricing.log")

PRICING_FILE = os.getenv("PRICING_FILE", "pricing.yaml")


def get_pricing_tiers() -> dict:
    if not os.path.exists(PRICING_FILE):
        logger.error(f"Pricing config not found at: {PRICING_FILE}")
        raise FileNotFoundError(f"Pricing config not found at: {PRICING_FILE}")

    with open(PRICING_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "tiers" not in config:
        logger.error(f"Pricing config at {PRICING_FILE} is missing a 'tiers' list.")
        raise ValueError(f"Pricing config at {PRICING_FILE} is missing a 'tiers' list.")

    return config
