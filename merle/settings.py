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

# Model Configuration
OLLAMA_MODEL_CONTEXT_WINDOW_SIZE = int(os.getenv("OLLAMA_MODEL_CONTEXT_WINDOW_SIZE", "2048"))

# Lambda Configuration Limits
LAMBDA_MEMORY_SIZE_MIN = 128  # MB - AWS Lambda minimum
LAMBDA_MEMORY_SIZE_MAX = 10240  # MB (10 GB) - AWS Lambda maximum
LAMBDA_MEMORY_SIZE_DEFAULT = 8192  # MB (8 GB) - Default memory allocation for 7B models


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


def validate_lambda_memory_size(memory_size: int) -> None:
    """
    Validate Lambda memory size is within AWS limits.

    Args:
        memory_size: Memory size in MB

    Raises:
        TypeError: If memory size is not an integer
        ValueError: If memory size is outside valid range
    """
    if not isinstance(memory_size, int):
        error_msg = f"Memory size must be an integer, got {type(memory_size).__name__}"
        raise TypeError(error_msg)

    if memory_size < LAMBDA_MEMORY_SIZE_MIN or memory_size > LAMBDA_MEMORY_SIZE_MAX:
        error_msg = (
            f"Lambda memory size must be between {LAMBDA_MEMORY_SIZE_MIN} MB and {LAMBDA_MEMORY_SIZE_MAX} MB. "
            f"Got: {memory_size} MB"
        )
        raise ValueError(error_msg)
