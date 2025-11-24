"""Initialize Ollama server for Lambda deployment."""

import logging
import os
import subprocess
import time
from pathlib import Path

import httpx

from merle import settings

logger = logging.getLogger(__name__)

# HTTP status code constant
HTTP_OK = 200


def start_ollama_server() -> subprocess.Popen | None:
    """
    Start Ollama server in the background.

    Returns:
        Popen object of the running server, or None if failed
    """
    try:
        # In Lambda, create hybrid setup: symlink blobs from /var/task, use /tmp for writable files
        if settings.is_lambda():
            import shutil  # noqa: PLC0415

            source_models = Path("/var/task/models")
            target_models = Path("/tmp/models")  # noqa: S108 - Lambda writable directory

            # Create /tmp/models structure if it doesn't exist
            if not target_models.exists():
                logger.info(f"Setting up hybrid models directory at {target_models}")
                target_models.mkdir(parents=True, exist_ok=True)

                # Symlink the large blobs directory (read-only, ~2GB+)
                blobs_src = source_models / "blobs"
                blobs_dst = target_models / "blobs"
                if blobs_src.exists() and not blobs_dst.exists():
                    logger.info(f"Symlinking blobs from {blobs_src} to {blobs_dst}")
                    blobs_dst.symlink_to(blobs_src)

                # Copy small manifests directory (writable, <1MB)
                manifests_src = source_models / "manifests"
                manifests_dst = target_models / "manifests"
                if manifests_src.exists() and not manifests_dst.exists():
                    logger.info(f"Copying manifests from {manifests_src} to {manifests_dst}")
                    shutil.copytree(manifests_src, manifests_dst)

                logger.info("Hybrid models directory setup complete")
        else:
            # Non-Lambda: ensure models directory exists
            settings.get_models_path().mkdir(parents=True, exist_ok=True)

        # Set environment for Ollama
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = settings.OLLAMA_MODELS_DIR
        env["OLLAMA_HOST"] = "0.0.0.0:11434"
        env["HOME"] = "/tmp"  # noqa: S108 - Ollama requires HOME to be set in Lambda

        logger.info("Starting Ollama server...")

        # Use absolute path to Ollama binary in /var/task/bin (required for Zappa ZIP deployment)
        # Zappa only packages files from /var/task/, so binary must be there
        ollama_binary = "/var/task/bin/ollama"
        if not Path(ollama_binary).exists():
            logger.error(f"Ollama binary not found at {ollama_binary}")
            raise FileNotFoundError(f"Ollama binary not found at {ollama_binary}")  # noqa: TRY301

        # Start Ollama server in background
        process = subprocess.Popen(  # noqa: S603 - ollama binary path is controlled
            [ollama_binary, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for server to be ready
        max_retries = settings.OLLAMA_STARTUP_TIMEOUT
        retry_delay = 1

        for i in range(max_retries):
            try:
                response = httpx.get(f"{settings.OLLAMA_URL}/api/tags", timeout=2.0)
                if response.status_code == HTTP_OK:
                    logger.info(f"Ollama server started successfully on {settings.OLLAMA_URL}")
                    return process
            except (httpx.ConnectError, httpx.TimeoutException):
                logger.debug(f"Waiting for Ollama server... ({i + 1}/{max_retries})")
                time.sleep(retry_delay)
        logger.error("Failed to start Ollama server: timeout waiting for response")
        process.kill()
        return None  # noqa: TRY300

    except FileNotFoundError:
        logger.exception("Ollama binary not found in PATH")
        return None
    except Exception:
        logger.exception("Error starting Ollama server")
        return None


def ensure_model_available(model: str | None = None) -> bool:
    """
    Ensure the specified model is available, downloading if necessary.

    Args:
        model: Model name to check/download

    Returns:
        True if model is available, False otherwise
    """
    model_name = model or settings.OLLAMA_MODEL

    try:
        logger.info(f"Checking availability of model: {model_name}")

        # Check if model exists
        response = httpx.get(f"{settings.OLLAMA_URL}/api/tags", timeout=10.0)
        response.raise_for_status()

        models_data = response.json()
        available_models = [m["name"] for m in models_data.get("models", [])]

        if any(model_name in m for m in available_models):
            logger.info(f"Model {model_name} is already available")
            return True

        # Model not found, attempt to pull it
        logger.info(f"Model {model_name} not found, attempting to pull...")

        pull_response = httpx.post(
            f"{settings.OLLAMA_URL}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=settings.OLLAMA_REQUEST_TIMEOUT,
        )
        pull_response.raise_for_status()

        logger.info(f"Successfully pulled model: {model_name}")
        return True  # noqa: TRY300

    except httpx.TimeoutException:
        logger.exception(f"Timeout while pulling model {model_name}")
        return False
    except httpx.HTTPError:
        logger.exception("HTTP error while ensuring model availability")
        return False
    except Exception:
        logger.exception("Error ensuring model availability")
        return False


def initialize_ollama() -> bool:
    """
    Initialize Ollama server and ensure model is available.

    This function should be called when the Lambda container starts.

    Returns:
        True if initialization successful, False otherwise
    """
    global _server_process  # noqa: PLW0603

    logger.info("Initializing Ollama for Lambda...")

    # Check if we should use external Ollama (skip initialization)
    if not settings.is_local_ollama():
        logger.info(f"Using external Ollama server at: {settings.OLLAMA_URL}")
        return True

    # Start local Ollama server
    _server_process = start_ollama_server()
    if not _server_process:
        logger.error("Failed to start Ollama server")
        return False

    # Ensure model is available
    if not ensure_model_available():
        logger.warning(f"Model {settings.OLLAMA_MODEL} is not available")
        return False

    logger.info("Ollama initialization completed successfully")
    return True


# Module-level state for singleton pattern
_initialized = False
_server_process: subprocess.Popen | None = None


def get_or_initialize() -> bool:
    """
    Get or initialize Ollama server (singleton pattern).

    Returns:
        True if Ollama is ready, False otherwise
    """
    global _initialized  # noqa: PLW0603

    if _initialized:
        return True

    try:
        # Check if server is already running
        response = httpx.get(f"{settings.OLLAMA_URL}/api/tags", timeout=2.0)
        if response.status_code == HTTP_OK:
            logger.info("Ollama server already running")
            _initialized = True
            return True
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Initialize Ollama
    success = initialize_ollama()
    _initialized = success
    return success
