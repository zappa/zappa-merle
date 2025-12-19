"""Tests for merle.chat module."""

import time
from io import StringIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from merle.chat import ChatClient, WaitingCursor, run_interactive_chat


class TestWaitingCursor:
    """Tests for WaitingCursor class."""

    def test_init(self):
        """Test WaitingCursor initialization."""
        cursor = WaitingCursor()
        assert cursor.running is False
        assert cursor.thread is None
        assert len(cursor.spinner_chars) > 0

    def test_start_and_stop(self):
        """Test starting and stopping the cursor."""
        cursor = WaitingCursor()
        with patch("sys.stdout", new_callable=StringIO):
            cursor.start()
            assert cursor.running is True
            assert cursor.thread is not None
            time.sleep(0.3)  # Let it spin a bit
            cursor.stop()
            assert cursor.running is False

    def test_stop_without_start(self):
        """Test that stopping without starting doesn't crash."""
        cursor = WaitingCursor()
        cursor.stop()  # Should not raise an exception
        assert cursor.running is False

    def test_multiple_starts(self):
        """Test that multiple starts don't create multiple threads."""
        cursor = WaitingCursor()
        with patch("sys.stdout", new_callable=StringIO):
            cursor.start()
            first_thread = cursor.thread
            cursor.start()  # Second start should be ignored
            assert cursor.thread == first_thread
            cursor.stop()


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
            response, metrics = client._make_request(messages)

        assert response == "Hello there!"
        assert isinstance(metrics, dict)

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
    def test_make_request_gateway_timeout(self, mock_client_class: MagicMock, capsys: pytest.CaptureFixture[str]):
        """Test request with 504 Gateway Timeout returns empty response with helpful message."""
        mock_response = MagicMock()
        mock_response.status_code = 504
        mock_response.text = "Gateway Timeout"

        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = None
        mock_stream.raise_for_status.side_effect = httpx.HTTPStatusError(
            "504 Gateway Timeout", request=MagicMock(), response=mock_response
        )

        mock_client.stream.return_value = mock_stream
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        mock_client_class.return_value = mock_client

        client = ChatClient("https://example.com", "token", "llama2")
        messages = [{"role": "user", "content": "Hello"}]

        # 504 errors should return empty response instead of raising exception
        response, metrics = client._make_request(messages)
        assert response == ""
        assert metrics == {"prompt_eval_count": 0, "eval_count": 0, "prompt_eval_duration": 0, "eval_duration": 0}

        # Check that the helpful message was printed
        captured = capsys.readouterr()
        assert "504 Gateway Timeout" in captured.out
        assert "cold start" in captured.out

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
            response, metrics = client._make_request(messages)

        # Should still return valid content, skipping invalid line
        assert response == "Valid"
        assert isinstance(metrics, dict)


class TestRunInteractiveChat:
    """Tests for run_interactive_chat function."""

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_exit_command(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
        """Test exiting chat with /exit command."""
        mock_input.side_effect = ["/exit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Verify client was created (context_window_size defaults to 2048 when not provided)
        mock_client_class.assert_called_once_with(
            "https://example.com", "token", "llama2", system_prompt=None, context_window_size=2048
        )

        # Verify chat was never called (exited immediately)
        mock_client.chat.assert_not_called()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_quit_command(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
        """Test exiting chat with /quit command."""
        mock_input.side_effect = ["/quit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        mock_client.chat.assert_not_called()

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_reset_command(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
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
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_sends_message(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
        """Test sending a chat message."""
        mock_input.side_effect = ["Hello!", "/exit"]
        mock_client = MagicMock()
        mock_client.chat.return_value = "Hi there!"
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Verify chat was called with user message and prompt
        mock_client.chat.assert_called_once_with("Hello!", prompt="[llama2]: ")

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_empty_input(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
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
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_keyboard_interrupt(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
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
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_http_error(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
        """Test handling of HTTP error during chat."""
        mock_input.side_effect = ["Hello", "/exit"]
        mock_client = MagicMock()
        mock_client.chat.side_effect = httpx.HTTPStatusError("401", request=MagicMock(), response=MagicMock())
        mock_client_class.return_value = mock_client

        with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2")

        # Should continue after error
        assert mock_input.call_count == 2

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_debug_flag_disabled(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
        """Test that debug=False sets logging to WARNING level."""
        mock_input.side_effect = ["/exit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2", debug=False)

        # Verify logging level was set to WARNING for merle and httpx
        assert mock_get_logger.call_count >= 2
        mock_get_logger.assert_any_call("merle")
        mock_get_logger.assert_any_call("httpx")
        # At least one call should set WARNING level
        assert any(call[0][0] == 30 for call in mock_logger.setLevel.call_args_list)  # 30 is WARNING level

    @patch("merle.chat.input")
    @patch("merle.chat.ChatClient")
    @patch("merle.chat.logging.getLogger")
    def test_interactive_chat_debug_flag_enabled(
        self, mock_get_logger: MagicMock, mock_client_class: MagicMock, mock_input: MagicMock
    ):
        """Test that debug=True sets logging to INFO level."""
        mock_input.side_effect = ["/exit"]
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        with patch("sys.stdout", new_callable=StringIO):
            run_interactive_chat("https://example.com", "token", "llama2", debug=True)

        # Verify logging level was set to INFO for merle and httpx
        assert mock_get_logger.call_count >= 2
        mock_get_logger.assert_any_call("merle")
        mock_get_logger.assert_any_call("httpx")
        # At least one call should set INFO level
        assert any(call[0][0] == 20 for call in mock_logger.setLevel.call_args_list)  # 20 is INFO level


class TestContextWindowManagement:
    """Tests for context window management and automatic trimming."""

    def test_init_with_context_window_size(self):
        """Test ChatClient initialization with context window size."""
        client = ChatClient(
            base_url="https://example.com",
            auth_token="token",
            model="llama2",
            context_window_size=4096,
        )

        assert client.context_window_size == 4096
        assert client.context_window_threshold == int(4096 * 0.8)  # 3276

    def test_init_default_context_window_size(self):
        """Test ChatClient uses default context window size when not specified."""
        client = ChatClient(
            base_url="https://example.com",
            auth_token="token",
            model="llama2",
        )

        assert client.context_window_size == 2048
        assert client.context_window_threshold == int(2048 * 0.8)  # 1638

    def test_conversation_summary_includes_context_window(self):
        """Test that conversation summary includes context window information."""
        client = ChatClient(
            base_url="https://example.com",
            auth_token="token",
            model="llama2",
            context_window_size=4096,
        )

        # Simulate token usage
        client.last_prompt_tokens = 1000

        summary = client.get_conversation_summary()

        assert summary["context_window_size"] == 4096
        assert summary["context_window_threshold"] == int(4096 * 0.8)
        assert summary["context_usage_percent"] == 24.4  # 1000/4096 * 100

    def test_context_usage_percent_zero_when_no_tokens(self):
        """Test that context usage percent is 0 when no tokens used."""
        client = ChatClient(
            base_url="https://example.com",
            auth_token="token",
            model="llama2",
            context_window_size=4096,
        )

        summary = client.get_conversation_summary()
        assert summary["context_usage_percent"] == 0.0

    @patch("merle.chat.httpx.Client")
    def test_trim_conversation_history_when_threshold_exceeded(self, mock_client_class: MagicMock):
        """Test that conversation history is trimmed when threshold is exceeded."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                '{"message": {"role": "assistant", "content": "Response"}}',
                '{"done": true, "prompt_eval_count": 2000, "eval_count": 50}',
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

        # Create client with small context window
        client = ChatClient("https://example.com", "token", "llama2", context_window_size=2048)

        # Add multiple messages to conversation history
        for i in range(10):
            client._add_message("user", f"Message {i}")
            client._add_message("assistant", f"Response {i}")

        assert len(client.conversation_history) == 20  # 10 user + 10 assistant

        # Make a request that exceeds threshold (mocked to return 2000 tokens)
        with patch("sys.stdout", new_callable=StringIO):
            client.chat("Test message")

        # Verify that messages were trimmed
        # Threshold is 1638 (80% of 2048), and we got 2000 tokens
        # Should have removed approximately 1/3 of messages (6-7 messages in pairs)
        assert len(client.conversation_history) < 20
        assert len(client.conversation_history) >= 10  # Should keep some messages

    @patch("merle.chat.httpx.Client")
    def test_trim_preserves_system_messages(self, mock_client_class: MagicMock):
        """Test that trimming preserves system messages."""
        # Mock successful response with high token count
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                '{"message": {"role": "assistant", "content": "Response"}}',
                '{"done": true, "prompt_eval_count": 2000, "eval_count": 50}',
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

        # Create client with system prompt and small context window
        client = ChatClient(
            "https://example.com",
            "token",
            "llama2",
            system_prompt="You are a helpful assistant",
            context_window_size=2048,
        )

        # Add multiple conversation messages
        for i in range(10):
            client._add_message("user", f"Message {i}")
            client._add_message("assistant", f"Response {i}")

        system_messages_before = len(client.get_messages_by_role("system"))
        assert system_messages_before == 1

        # Make a request that exceeds threshold
        with patch("sys.stdout", new_callable=StringIO):
            client.chat("Test message")

        # Verify system messages are still present
        system_messages_after = len(client.get_messages_by_role("system"))
        assert system_messages_after == system_messages_before

    def test_no_trim_when_below_threshold(self):
        """Test that no trimming occurs when below threshold."""
        client = ChatClient("https://example.com", "token", "llama2", context_window_size=4096)

        # Add a few messages
        client._add_message("user", "Message 1")
        client._add_message("assistant", "Response 1")
        client._add_message("user", "Message 2")
        client._add_message("assistant", "Response 2")

        # Set token count below threshold
        client.last_prompt_tokens = 1000  # Well below 3276 threshold

        messages_before = len(client.conversation_history)

        # Call trim directly
        client._trim_conversation_history()

        # Verify no messages were removed
        assert len(client.conversation_history) == messages_before
