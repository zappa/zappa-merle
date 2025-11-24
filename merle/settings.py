"""Configuration management for Merle Ollama Model Server."""

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# logging.basicConfig set in __init__.py
logger = logging.getLogger()

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
if LOG_LEVEL and LOG_LEVEL in ("INFO", "ERROR", "WARNING", "DEBUG", "CRITICAL"):
    level = getattr(logging, LOG_LEVEL)
    logger.setLevel(level)

DEFAULT_REGION = "ap-northeast-1"
REGION = os.getenv("REGION", DEFAULT_REGION)

DEFAULT_STAGE = "dev"
STAGE = os.getenv("STAGE", DEFAULT_STAGE)

# Ollama Server Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", OLLAMA_HOST)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_MODELS_DIR = os.getenv("OLLAMA_MODELS", "/var/task/models")

# Lambda Configuration
AWS_LAMBDA_FUNCTION_NAME = os.getenv("AWS_LAMBDA_FUNCTION_NAME")

# Zappa Configuration
ZAPPA_RUNNING_IN_DOCKER = os.getenv("ZAPPA_RUNNING_IN_DOCKER", "False").lower() == "true"

# Timeouts and Limits
OLLAMA_STARTUP_TIMEOUT = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "30"))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300.0"))


def get_models_path() -> Path:
    """Get the models directory as a Path object."""
    return Path(OLLAMA_MODELS_DIR)


def is_local_ollama() -> bool:
    """Check if Ollama is running locally vs external server."""
    host = OLLAMA_URL.lower()
    return "localhost" in host or "127.0.0.1" in host


def is_lambda() -> bool:
    """Check if running in AWS Lambda environment."""
    return AWS_LAMBDA_FUNCTION_NAME is not None
