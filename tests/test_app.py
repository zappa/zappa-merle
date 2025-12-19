"""Tests for merle.app Flask application."""

import json
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from merle.app import app, estimate_token_count


class TestEstimateTokenCount:
    """Tests for token estimation function."""

    def test_empty_messages(self):
        """Test token count for empty message list."""
        result = estimate_token_count([])
        assert result == 0

    def test_single_simple_message(self):
        """Test token count for a single simple message."""
        messages = [{"role": "user", "content": "Hello world"}]
        result = estimate_token_count(messages)
        # "user" (4 chars) + "Hello world" (11 chars) = 15 chars
        # 15 / 4 * 1.1 = 4.125 -> 4 tokens
        assert result == 4

    def test_multiple_messages(self):
        """Test token count for multiple messages."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]
        result = estimate_token_count(messages)
        # Total chars: 4 + 15 + 9 + 34 + 4 + 13 = 79
        # 79 / 4 * 1.1 = 21.725 -> 21 tokens
        assert result > 0
        assert isinstance(result, int)

    def test_long_message(self):
        """Test token count for a very long message."""
        long_content = "a" * 4000  # 4000 characters
        messages = [{"role": "user", "content": long_content}]
        result = estimate_token_count(messages)
        # "user" (4 chars) + 4000 chars = 4004 chars
        # 4004 / 4 * 1.1 = 1101.1 -> 1101 tokens
        assert result > 1000
        assert result < 1200

    def test_multimodal_content(self):
        """Test token count for multimodal content with text and images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image", "url": "https://example.com/image.jpg"},
                ],
            }
        ]
        result = estimate_token_count(messages)
        # Only text is counted: "user" (4) + "What do you see in this image?" (32) = 36 chars
        # 36 / 4 * 1.1 = 9.9 -> 9 tokens
        assert result > 0

    def test_missing_content_field(self):
        """Test handling of messages with missing content field."""
        messages = [{"role": "user"}]
        result = estimate_token_count(messages)
        # Only "user" (4 chars) = 1.1 -> 1 token
        assert result >= 0


class TestAppContextWindowValidation:
    """Tests for Flask app context window validation."""

    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    @pytest.fixture(autouse=True)
    def mock_ollama_init(self):
        """Mock Ollama initialization to avoid errors in tests."""
        with patch("merle.app.get_or_initialize", return_value=True):
            yield

    def test_chat_request_within_context_window(self, client: MagicMock):
        """Test that requests within context window are allowed."""
        # Mock httpx client to avoid actual requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read.return_value = b'{"response": "Hello!"}'

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client_instance = MagicMock()
        mock_client_instance.stream.return_value = mock_stream
        mock_client_instance.close.return_value = None

        with (
            patch("merle.app.httpx.Client", return_value=mock_client_instance),
            patch("merle.app.settings.OLLAMA_MODEL_CONTEXT_WINDOW_SIZE", 2048),
        ):
            # Small request that should be within context window
            response = client.post(
                "/api/chat",
                data=json.dumps(
                    {
                        "model": "llama2",
                        "messages": [
                            {"role": "user", "content": "Hello"},
                        ],
                        "stream": False,
                    }
                ),
                content_type="application/json",
            )

            # Should proxy to Ollama successfully
            assert response.status_code == 200

    def test_chat_request_exceeds_context_window(self, client: MagicMock):
        """Test that requests exceeding context window return 413."""
        with patch("merle.app.settings.OLLAMA_MODEL_CONTEXT_WINDOW_SIZE", 10):
            # Large request that exceeds small context window
            long_content = "a" * 1000  # Will be ~275 tokens (far exceeds 10)
            response = client.post(
                "/api/chat",
                data=json.dumps(
                    {
                        "model": "llama2",
                        "messages": [
                            {"role": "user", "content": long_content},
                        ],
                        "stream": False,
                    }
                ),
                content_type="application/json",
            )

            # Should return 413 Payload Too Large
            assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            data = json.loads(response.data)
            assert "error" in data
            assert "exceeds model context window" in data["error"].lower()
            assert "estimated_tokens" in data

    def test_chat_request_at_boundary(self, client: MagicMock):
        """Test request exactly at context window boundary."""
        # Mock httpx to avoid actual requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read.return_value = b'{"response": "OK"}'

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client_instance = MagicMock()
        mock_client_instance.stream.return_value = mock_stream
        mock_client_instance.close.return_value = None

        with (
            patch("merle.app.httpx.Client", return_value=mock_client_instance),
            patch("merle.app.settings.OLLAMA_MODEL_CONTEXT_WINDOW_SIZE", 100),
        ):
            # Request with approximately 90 tokens (within limit)
            content = "a" * 320  # ~88 tokens with overhead
            response = client.post(
                "/api/chat",
                data=json.dumps(
                    {
                        "model": "llama2",
                        "messages": [
                            {"role": "user", "content": content},
                        ],
                        "stream": False,
                    }
                ),
                content_type="application/json",
            )

            # Should be allowed
            assert response.status_code == 200

    def test_non_chat_endpoint_not_validated(self, client: MagicMock):
        """Test that non-chat endpoints are not validated."""
        # Mock httpx client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read.return_value = b'{"models": []}'

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client_instance = MagicMock()
        mock_client_instance.stream.return_value = mock_stream
        mock_client_instance.close.return_value = None

        with (
            patch("merle.app.httpx.Client", return_value=mock_client_instance),
            patch("merle.app.settings.OLLAMA_MODEL_CONTEXT_WINDOW_SIZE", 10),
        ):
            # Request to /api/tags endpoint (not chat)
            response = client.get("/api/tags")

            # Should not be blocked by context window validation
            assert response.status_code == 200

    def test_chat_request_without_messages(self, client: MagicMock):
        """Test chat request without messages field."""
        # Mock httpx client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read.return_value = b'{"response": "OK"}'

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client_instance = MagicMock()
        mock_client_instance.stream.return_value = mock_stream
        mock_client_instance.close.return_value = None

        with (
            patch("merle.app.httpx.Client", return_value=mock_client_instance),
            patch("merle.app.settings.OLLAMA_MODEL_CONTEXT_WINDOW_SIZE", 2048),
        ):
            # Request without messages field
            response = client.post(
                "/api/chat",
                data=json.dumps(
                    {
                        "model": "llama2",
                        "stream": False,
                    }
                ),
                content_type="application/json",
            )

            # Should not be blocked (no messages to validate)
            assert response.status_code == 200

    def test_health_endpoint(self, client: MagicMock):
        """Test health check endpoint."""
        # Mock httpx response for health check
        with patch("merle.app.httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = client.get("/health")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["status"] == "healthy"

    def test_root_endpoint(self, client: MagicMock):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["service"] == "merle-ollama-proxy"
