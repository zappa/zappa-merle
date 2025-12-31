"""Pytest fixtures for testing merle."""

from pathlib import Path

import pytest


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """
    Create a temporary cache directory for testing.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to temporary cache directory
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_models_dir(tmp_path: Path) -> Path:
    """
    Create a mock Ollama models directory for testing.

    Creates an empty models directory structure to avoid tests
    using the real system Ollama models.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to mock models directory
    """
    models_dir = tmp_path / ".ollama" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


@pytest.fixture
def sample_config() -> dict:
    """
    Return a sample configuration dictionary.

    Returns:
        Sample config dict
    """
    return {
        "models": {
            "llama2": {
                "dev": {
                    "normalized_name": "llama2",
                    "cache_dir": "dev/llama2",
                    "auth_token": "test-token-123",
                    "region": "us-east-1",
                    "last_updated": "2025-01-01T00:00:00Z",
                }
            },
            "mistral": {
                "dev": {
                    "normalized_name": "mistral",
                    "cache_dir": "dev/mistral",
                    "auth_token": "test-token-456",
                    "region": "us-west-2",
                    "tags": {"Environment": "dev", "Project": "test"},
                    "last_updated": "2025-01-01T00:00:00Z",
                }
            },
        }
    }


@pytest.fixture
def sample_tags() -> dict[str, str]:
    """
    Return sample AWS tags.

    Returns:
        Sample tags dict
    """
    return {
        "Environment": "dev",
        "Project": "ollama",
        "Owner": "test-user",
    }


@pytest.fixture
def mock_ollama_api_response() -> dict:
    """
    Return a mock Ollama API response.

    Returns:
        Mock API response dict
    """
    return {
        "models": [
            {"name": "llama2:latest"},
            {"name": "mistral:latest"},
            {"name": "gemma:latest"},
        ]
    }


@pytest.fixture
def mock_chat_response() -> str:
    """
    Return a mock chat response.

    Returns:
        Mock chat response as newline-delimited JSON
    """
    return "\n".join(
        [
            '{"message": {"role": "assistant", "content": "Hello"}}',
            '{"message": {"role": "assistant", "content": " there"}}',
            '{"message": {"role": "assistant", "content": "!"}}',
            '{"done": true}',
        ]
    )
