"""Tests for merle.chat module."""

from io import StringIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from merle.chat import ChatClient, run_interactive_chat


class TestChatClient:
    """Tests for ChatClient class."""

    def test_init(self):
        """Test ChatClient initialization."""
        client = ChatClient(
            base_url="https://example.com/api",
            auth_token="test-token",
            model="llama2",
        )

        assert client.base_url == "https://example.com/api"
        assert client.auth_token == "test-token"
        assert client.model == "llama2"
        assert client.conversation_history == []

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from base_url."""
        client = ChatClient(
            base_url="https://example.com/",
            auth_token="test-token",
            model="llama2",
        )

        assert client.base_url == "https://example.com"

    def test_reset(self):
        """Test resetting conversation history."""
        client = ChatClient("https://example.com", "token", "llama2")
        client.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        client.reset()

        assert client.conversation_history == []

    @patch("merle.chat.httpx.Client")
    def test_make_request_success(self, mock_client_class: MagicMock):
        """Test successful streaming request."""
        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Simulate streaming lines
        mock_lines = [
            '{"message": {"role": "assistant", "content": "Hello"}}',
            '{"message": {"role": "assistant", "content": " there"}}',
            '{"message": {"role": "assistant", "content": "!"}}',
            '{"done": true}',
        ]
        mock_response.iter_lines.return_value = iter(mock_lines)
        mock_response.raise_for_status.return_value = None

        # Mock the context manager
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("sys.stdout", new_callable=StringIO):
            result = client._make_request(messages)

        assert result == "Hello there!"

    @patch("merle.chat.httpx.Client")
    def test_make_request_http_error(self, mock_client_class: MagicMock):
        """Test request with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(httpx.HTTPStatusError):
            client._make_request(messages)

    @patch("merle.chat.httpx.Client")
    def test_make_request_connection_error(self, mock_client_class: MagicMock):
        """Test request with connection error."""
        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.ConnectError("Connection failed")
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(httpx.ConnectError):
            client._make_request(messages)

    @patch("merle.chat.httpx.Client")
    def test_make_request_timeout(self, mock_client_class: MagicMock):
        """Test request with timeout."""
        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.TimeoutException("Request timeout")
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(httpx.TimeoutException):
            client._make_request(messages)

    @patch("merle.chat.httpx.Client")
    def test_chat_adds_to_history(self, mock_client_class: MagicMock):
        """Test that chat adds messages to conversation history."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                '{"message": {"role": "assistant", "content": "Hi!"}}',
                '{"done": true}',
            ]
        )
        mock_response.raise_for_status.return_value = None

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")

        with patch("sys.stdout", new_callable=StringIO):
            response = client.chat("Hello")

        assert response == "Hi!"
        assert len(client.conversation_history) == 2
        assert client.conversation_history[0] == {"role": "user", "content": "Hello"}
        assert client.conversation_history[1] == {"role": "assistant", "content": "Hi!"}

    @patch("merle.chat.httpx.Client")
    def test_make_request_invalid_json_line(self, mock_client_class: MagicMock):
        """Test handling of invalid JSON in streaming response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                '{"message": {"role": "assistant", "content": "Valid"}}',
                "invalid json line",
                '{"done": true}',
            ]
        )
        mock_response.raise_for_status.return_value = None

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("sys.stdout", new_callable=StringIO):
            result = client._make_request(messages)

        # Should still return valid content, skipping invalid line
        assert result == "Valid"


class TestRunInteractiveChat:
    """Tests for run_interactive_chat function."""

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_exit_command(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test exiting chat with /exit command."""
        mock_input.side_effect = ["/exit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Verify client was created
        mock_client_class.assert_called_once_with("https://example.com", "token", "llama2")

        # Verify chat was never called (exited immediately)
        mock_client.chat.assert_not_called()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_quit_command(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test exiting chat with /quit command."""
        mock_input.side_effect = ["/quit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        mock_client.chat.assert_not_called()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_reset_command(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test resetting conversation with /reset command."""
        mock_input.side_effect = ["/reset", "/exit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Verify reset was called
        mock_client.reset.assert_called_once()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_sends_message(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test sending a chat message."""
        mock_input.side_effect = ["Hello!", "/exit"]
        mock_client = MagicMock()
        mock_client.chat.return_value = "Hi there!"
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Verify chat was called with user message
        mock_client.chat.assert_called_once_with("Hello!")

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_empty_input(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test that empty input is skipped."""
        mock_input.side_effect = ["", "  ", "/exit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Verify chat was never called (only empty inputs)
        mock_client.chat.assert_not_called()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_keyboard_interrupt(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test handling of KeyboardInterrupt."""
        mock_input.side_effect = KeyboardInterrupt()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Should exit gracefully without calling chat
        mock_client.chat.assert_not_called()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    def test_interactive_chat_http_error(self, mock_client_class: MagicMock, mock_input: MagicMock):
        """Test handling of HTTP error during chat."""
        mock_input.side_effect = ["Hello", "/exit"]
        mock_client = MagicMock()
        mock_client.chat.side_effect = httpx.HTTPStatusError("401", request=MagicMock(), response=MagicMock())
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Should continue after error
        assert mock_input.call_count == 2
