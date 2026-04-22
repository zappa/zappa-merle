"""Flask application for Ollama model server proxy."""

import contextlib
import logging
from http import HTTPStatus
from typing import Any

import httpx
from flask import Flask, Response, request, stream_with_context
from werkzeug.exceptions import HTTPException

from merle import settings
from merle.init_ollama import get_or_initialize

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """
    Estimate token count for a list of messages.

    Uses a simple heuristic: ~4 characters per token.
    This is approximate but sufficient for context window validation.

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        # Count role
        total_chars += len(msg.get("role", ""))
        # Count content
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Handle multimodal content (text + images)
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total_chars += len(item["text"])

    # Approximate: 4 chars per token (industry standard approximation)
    # Add 10% buffer for formatting/structure overhead
    estimated_tokens = int((total_chars / 4) * 1.1)
    return estimated_tokens


# Initialize Flask app
app = Flask(__name__)

# Initialize Ollama server on app startup (for Lambda cold start)
logger.info("Initializing Ollama server...")
if not get_or_initialize():
    logger.error("Failed to initialize Ollama server")
else:
    logger.info("Ollama server initialized successfully")


@app.before_request
def require_api_key() -> tuple[dict[str, str], int] | None:
    """
    Enforce X-API-Key on every request when MERLE_REQUIRE_API_KEY is enabled.

    The function-url topology exposes Lambda with AuthType=NONE, so the app itself
    must validate the token. The apigw topology disables this gate because the
    API Gateway authorizer already rejects unauthenticated traffic upstream.
    """
    if not settings.MERLE_REQUIRE_API_KEY:
        return None
    if not settings.API_KEY:
        logger.error("MERLE_REQUIRE_API_KEY is set but API_KEY is not configured")
        return {"error": "Server misconfigured"}, HTTPStatus.INTERNAL_SERVER_ERROR
    provided = request.headers.get("X-API-Key", "")
    if provided != settings.API_KEY:
        logger.warning("Rejected request with missing or invalid X-API-Key")
        return {"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED
    return None


@app.route("/health", methods=["GET"])
def health_check() -> tuple[dict[str, Any], int]:
    """Health check endpoint."""
    try:
        # Check if Ollama is responding
        response = httpx.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5.0)
        if response.status_code == HTTPStatus.OK:
            return {"status": "healthy", "model": settings.OLLAMA_MODEL}, HTTPStatus.OK
    except Exception:
        logger.exception("Health check failed")

    return {"status": "unhealthy", "error": "Internal error"}, HTTPStatus.SERVICE_UNAVAILABLE


def _validate_chat_context_window(data: Any) -> tuple[dict[str, Any], int] | None:  # noqa: ANN401
    """Reject chat requests whose messages exceed the configured context window."""
    if not (data and isinstance(data, dict) and "messages" in data):
        return None
    messages = data.get("messages", [])
    if not messages:
        return None
    estimated_tokens = estimate_token_count(messages)
    context_window_size = settings.OLLAMA_MODEL_CONTEXT_WINDOW_SIZE
    logger.info(
        f"Chat request validation - estimated tokens: {estimated_tokens}, context window: {context_window_size}"
    )
    if estimated_tokens > context_window_size:
        error_msg = (
            f"Request exceeds model context window. "
            f"Estimated tokens: {estimated_tokens}, "
            f"max context window: {context_window_size}. "
            f"Please reduce the conversation history or message length."
        )
        logger.warning(error_msg)
        return {
            "error": error_msg,
            "estimated_tokens": estimated_tokens,
        }, HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    return None


def _proxy(target_url: str) -> Response | tuple[dict[str, str], int]:  # noqa: C901, PLR0911, PLR0912, PLR0915
    """
    Forward the current Flask request to `target_url` on the local Ollama server.

    Shared by /api/* (Ollama-native) and /v1/* (OpenAI-compatible) proxies.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Get request data
        data = None
        if request.method in ["POST", "PUT"]:
            data = request.get_json(silent=True)

        # Get query parameters
        params = request.args.to_dict()

        # Forward headers (excluding host)
        headers = {
            key: value for key, value in request.headers.items() if key.lower() not in ["host", "content-length"]
        }

        logger.info(f"Proxying {request.method} request to {target_url}")

        # Check if client expects streaming response (from request body)
        is_streaming_request = False
        if data and isinstance(data, dict):
            is_streaming_request = data.get("stream", False)

        logger.info(f"Client requested streaming: {is_streaming_request}")

        # Use httpx streaming to get chunks as they arrive from Ollama
        client = httpx.Client(timeout=settings.OLLAMA_REQUEST_TIMEOUT)

        try:
            # Make streaming request to Ollama
            if request.method == "GET":
                stream_response = client.stream("GET", target_url, params=params, headers=headers)
            elif request.method == "POST":
                stream_response = client.stream("POST", target_url, json=data, params=params, headers=headers)
            elif request.method == "PUT":
                stream_response = client.stream("PUT", target_url, json=data, params=params, headers=headers)
            elif request.method == "DELETE":
                stream_response = client.stream("DELETE", target_url, params=params, headers=headers)
            else:
                client.close()
                return {"error": "Method not allowed"}, HTTPStatus.METHOD_NOT_ALLOWED

            http_response = stream_response.__enter__()
            logger.info(f"Got response with status {http_response.status_code}")

            content_type = http_response.headers.get("content-type", "")
            should_stream = is_streaming_request or "stream" in content_type or "ndjson" in content_type

            if should_stream:
                logger.info("Streaming response from Ollama")

                def generate():  # noqa: ANN202
                    try:
                        yield from http_response.iter_bytes()
                    finally:
                        stream_response.__exit__(None, None, None)
                        client.close()

                return Response(
                    stream_with_context(generate()),
                    status=http_response.status_code,
                    headers=dict(http_response.headers),
                )

            content = http_response.read()
            response_headers = dict(http_response.headers)
            status_code = http_response.status_code

            stream_response.__exit__(None, None, None)
            client.close()

            return Response(content, status=status_code, headers=response_headers)

        except Exception:  # noqa: BLE001
            with contextlib.suppress(Exception):
                client.close()
            raise

    except httpx.TimeoutException:
        logger.exception("Request to Ollama timed out")
        return {"error": "Request timed out"}, HTTPStatus.GATEWAY_TIMEOUT
    except httpx.ConnectError:
        logger.exception("Failed to connect to Ollama")
        return {"error": "Ollama server not available"}, HTTPStatus.SERVICE_UNAVAILABLE
    except Exception:
        logger.exception("Error proxying request")
        return {"error": "Internal server error"}, HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_to_ollama(path: str) -> Response | tuple[dict[str, str], int]:
    """Proxy all /api/* requests to Ollama's native API."""
    if path == "chat" and request.method in ["POST", "PUT"]:
        rejection = _validate_chat_context_window(request.get_json(silent=True))
        if rejection is not None:
            return rejection
    return _proxy(f"{settings.OLLAMA_URL}/api/{path}")


@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_to_ollama_openai(path: str) -> Response | tuple[dict[str, str], int]:
    """
    Proxy /v1/* requests to Ollama's OpenAI-compatible surface.

    Lets consumers point the `openai` SDK (or any OpenAI-compatible client) directly
    at the merle deployment without a client-side adapter.
    """
    if path == "chat/completions" and request.method in ["POST", "PUT"]:
        rejection = _validate_chat_context_window(request.get_json(silent=True))
        if rejection is not None:
            return rejection
    return _proxy(f"{settings.OLLAMA_URL}/v1/{path}")


@app.route("/", methods=["GET"])
def root() -> tuple[dict[str, Any], int]:
    """Root endpoint with service information."""
    return {
        "service": "merle-ollama-proxy",
        "description": "Ollama REST API proxy for AWS Lambda",
        "model": settings.OLLAMA_MODEL,
        "endpoints": {
            "/health": "Health check",
            "/api/*": "Ollama native API (proxied)",
            "/v1/*": "OpenAI-compatible API (proxied)",
        },
    }, HTTPStatus.OK


@app.errorhandler(HTTPStatus.NOT_FOUND)
def not_found(_error: HTTPException) -> tuple[dict[str, str], int]:
    """Handle 404 errors."""
    return {
        "error": "Endpoint not found. Use /api/* for Ollama native API or /v1/* for OpenAI-compatible API.",
    }, HTTPStatus.NOT_FOUND


@app.errorhandler(HTTPStatus.INTERNAL_SERVER_ERROR)
def internal_error(_error: HTTPException) -> tuple[dict[str, str], int]:
    """Handle 500 errors."""
    logger.exception("Internal server error")
    return {"error": "Internal server error"}, HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    # For local development only - not used in production Lambda deployment
    # The 0.0.0.0 binding and debug mode are acceptable for local dev
    app.run(host="0.0.0.0", port=8080, debug=True)  # noqa: S104, S201
