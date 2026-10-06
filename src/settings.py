"""Load shared project configuration from config.yaml."""

from pathlib import Path

from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    """Read the project configuration and validate its required sections."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if "paths" not in config or "retrieval" not in config:
        raise ValueError("config.yaml must contain 'paths' and 'retrieval' sections")

    return config


CONFIG = load_config()

PDF_DIR = PROJECT_ROOT / CONFIG["paths"]["pdf_dir"]
PARSED_DIR = PROJECT_ROOT / CONFIG["paths"]["parsed_dir"]
CHROMA_DIR = PROJECT_ROOT / CONFIG["paths"]["chroma_dir"]

COLLECTION_NAME = CONFIG["retrieval"]["collection_name"]
EMBEDDING_MODEL = CONFIG["retrieval"]["embedding_model"]
RETRIEVAL_MAX_DISTANCE = CONFIG["retrieval"].get("max_distance", 0.85)
