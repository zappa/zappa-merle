"""Chat client for interacting with deployed Ollama models."""

import json
import logging
import readline
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class WaitingCursor:
    """Simple waiting cursor that displays an animated spinner."""

    def __init__(self) -> None:
        """Initialize the waiting cursor."""
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.running = False
        self.thread: threading.Thread | None = None

    def _spin(self) -> None:
        """Run the spinner animation."""
        idx = 0
        while self.running:
            # Print spinner character and flush
            sys.stdout.write(f"\r{self.spinner_chars[idx]} ")
            sys.stdout.flush()
            idx = (idx + 1) % len(self.spinner_chars)
            time.sleep(0.1)
        # Clear the spinner when done
        sys.stdout.write("\r  \r")
        sys.stdout.flush()

    def start(self) -> None:
        """Start the spinner."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the spinner."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.5)


class ChatClient:
    """Client for chatting with a deployed Ollama model."""

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        model: str,
        system_prompt: str | None = None,
        context_window_size: int | None = None,
    ) -> None:
        """
        Initialize the chat client.

        Args:
            base_url: Base URL of the deployed API (e.g., https://xxx.execute-api.us-east-1.amazonaws.com)
            auth_token: Authentication token for API access
            model: Model name to use for chat
            system_prompt: Optional system prompt to initialize conversation context
            context_window_size: Maximum context window size in tokens (default: 2048)
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.model = model
        self.conversation_history: list[dict[str, Any]] = []

        # Context window management
        self.context_window_size = context_window_size or 2048
        self.context_window_threshold = int(self.context_window_size * 0.8)  # Trim at 80% to leave room

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        # Initialize conversation history with system prompt if provided
        if system_prompt:
            self._add_message("system", system_prompt)
            logger.info(f"Initialized conversation with system prompt ({len(system_prompt)} characters)")

    def _add_message(self, role: str, content: str, **kwargs: Any) -> None:  # noqa: ANN401
        """
        Add a message to the conversation history.

        Args:
            role: Message role (system, user, assistant, or tool)
            content: Message content
            **kwargs: Additional optional fields (e.g., images, tool_calls)
        """
        message: dict[str, Any] = {"role": role, "content": content}
        message.update(kwargs)
        self.conversation_history.append(message)
        logger.debug(f"Added {role} message to history (total messages: {len(self.conversation_history)})")

    def get_message_count(self) -> int:
        """
        Get the total number of messages in the conversation history.

        Returns:
            Number of messages
        """
        return len(self.conversation_history)

    def get_messages_by_role(self, role: str) -> list[dict[str, Any]]:
        """
        Get all messages with a specific role.

        Args:
            role: Message role to filter by

        Returns:
            List of messages with the specified role
        """
        return [msg for msg in self.conversation_history if msg.get("role") == role]

    def _trim_conversation_history(self) -> None:
        """
        Trim conversation history when it exceeds the context window threshold.

        Removes oldest user/assistant message pairs while preserving system messages.
        This is called automatically when the last prompt token count exceeds the threshold.
        """
        if self.last_prompt_tokens <= self.context_window_threshold:
            return

        # Separate system messages from conversation messages
        system_messages = [msg for msg in self.conversation_history if msg.get("role") == "system"]
        conversation_messages = [msg for msg in self.conversation_history if msg.get("role") != "system"]

        # Calculate how many messages to remove (remove in pairs when possible)
        messages_before = len(self.conversation_history)

        # Remove oldest messages until we're well below threshold
        # Aim for 60% of context window to leave plenty of room
        # Use a simple heuristic: if we're over threshold, remove oldest 1/3 of conversation
        messages_to_remove = max(2, len(conversation_messages) // 3)

        # Ensure we remove in pairs (user + assistant) when possible
        if messages_to_remove % 2 == 1 and messages_to_remove < len(conversation_messages):
            messages_to_remove += 1

        # Keep the most recent messages
        trimmed_conversation = conversation_messages[messages_to_remove:]

        # Rebuild conversation history with system messages first
        self.conversation_history = system_messages + trimmed_conversation

        messages_removed = messages_before - len(self.conversation_history)
        logger.warning(
            f"Context window threshold exceeded ({self.last_prompt_tokens} > {self.context_window_threshold} tokens). "
            f"Removed {messages_removed} oldest messages from conversation history. "
            f"Kept {len(self.conversation_history)} messages ({len(system_messages)} system, "
            f"{len(trimmed_conversation)} conversation)."
        )

    def _process_stream_line(
        self, line: str, spinner: WaitingCursor, first_content: bool, prompt: str | None = None
    ) -> tuple[str, bool, dict[str, Any] | None]:
        """
        Process a single line from the streaming response.

        Args:
            line: JSON line from the stream
            spinner: WaitingCursor instance to stop on first content
            first_content: Whether this is the first content received
            prompt: Optional prompt to display before first content (e.g., "[model]: ")

        Returns:
            Tuple of (content, is_first_content, metrics)
            metrics is a dict with token counts when done=true, None otherwise
        """
        try:
            data = json.loads(line)
            if "message" in data:
                content = data["message"].get("content", "")
                if content:
                    # Stop spinner and show prompt on first content
                    if first_content:
                        spinner.stop()
                        if prompt:
                            print(prompt, end="", flush=True)
                        first_content = False
                    print(content, end="", flush=True)
                    return content, first_content, None

            # Capture metrics when response is complete
            if data.get("done", False):
                metrics = {
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0),
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                    "eval_duration": data.get("eval_duration", 0),
                }
                logger.debug(
                    f"Token metrics - prompt: {metrics['prompt_eval_count']}, completion: {metrics['eval_count']}"
                )
                return "", first_content, metrics
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON: {line}")

        return "", first_content, None

    def _make_request(self, messages: list[dict[str, Any]], prompt: str | None = None) -> tuple[str, dict[str, Any]]:
        """
        Make a streaming request to the Ollama API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            prompt: Optional prompt to display before first content (e.g., "[model]: ")

        Returns:
            Tuple of (complete response text, metrics dict with token counts)

        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}/api/chat"
        headers = {
            "X-API-Key": self.auth_token,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Payload: {payload}")

        full_response = ""
        response_metrics: dict[str, Any] = {
            "prompt_eval_count": 0,
            "eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_duration": 0,
        }
        spinner = WaitingCursor()

        try:
            spinner.start()
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()

                    first_content = True
                    for line in response.iter_lines():
                        if not line:
                            continue

                        content, first_content, metrics = self._process_stream_line(
                            line, spinner, first_content, prompt
                        )
                        full_response += content

                        # Capture metrics when available
                        if metrics:
                            response_metrics = metrics

                        # Check if done
                        try:
                            if json.loads(line).get("done", False):
                                break
                        except json.JSONDecodeError:
                            pass

                spinner.stop()
                return full_response, response_metrics

        except httpx.HTTPStatusError as e:
            spinner.stop()
            if e.response.status_code == HTTPStatus.GATEWAY_TIMEOUT:
                error_msg = "[ERROR] -- 504 Gateway Timeout: If first request, this may be a cold start, try again"
                print(f"\n{error_msg}\n")
                # Return empty response to allow chat to continue
                return "", response_metrics
            error_body = self._extract_error_body(e)
            error_msg = f"HTTP error {e.response.status_code}: {error_body}"
            logger.error(error_msg)
            raise
        except httpx.ConnectError as e:
            spinner.stop()
            error_msg = f"Connection error: {e}"
            logger.error(error_msg)
            raise
        except httpx.TimeoutException as e:
            spinner.stop()
            error_msg = f"Request timeout: {e}"
            logger.error(error_msg)
            raise

    def _extract_error_body(self, error: httpx.HTTPStatusError) -> str:
        """
        Extract error body from HTTP error response.

        Args:
            error: HTTPStatusError from httpx

        Returns:
            Error body as string
        """
        try:
            # Try to get text property first (cached if already read)
            return error.response.text
        except (UnicodeDecodeError, AttributeError, OSError, httpx.StreamClosed):
            # If stream is closed or can't be read, return generic message
            return f"<unable to read response body - status {error.response.status_code}>"

    def chat(self, user_message: str, prompt: str | None = None) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: User's message
            prompt: Optional prompt to display before first content (e.g., "[model]: ")

        Returns:
            Assistant's response
        """
        # Add user message to history
        self._add_message("user", user_message)

        # Get response and metrics
        response, metrics = self._make_request(self.conversation_history, prompt=prompt)

        # Update token counters
        self.last_prompt_tokens = metrics.get("prompt_eval_count", 0)
        self.last_completion_tokens = metrics.get("eval_count", 0)
        self.total_prompt_tokens += self.last_prompt_tokens
        self.total_completion_tokens += self.last_completion_tokens
        self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        logger.debug(
            f"Token usage - prompt: {self.last_prompt_tokens}, completion: {self.last_completion_tokens}, "
            f"total session: {self.total_tokens}"
        )

        # Add assistant response to history
        self._add_message("assistant", response)

        # Trim conversation history if we exceeded the context window threshold
        self._trim_conversation_history()

        return response

    def reset(self) -> None:
        """
        Reset the conversation history and token counters, preserving system messages.

        This clears all user and assistant messages but keeps system messages
        to maintain the conversation context/instructions. Token counters are also reset.
        """
        # Preserve system messages
        system_messages = self.get_messages_by_role("system")
        message_count_before = len(self.conversation_history)
        self.conversation_history = system_messages
        messages_removed = message_count_before - len(system_messages)

        # Reset token counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        logger.info(
            f"Conversation history reset (removed {messages_removed} messages, "
            f"kept {len(system_messages)} system messages), token counters reset"
        )

    def get_conversation_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current conversation state.

        Returns:
            Dictionary with conversation statistics and token usage
        """
        return {
            "total_messages": len(self.conversation_history),
            "system_messages": len(self.get_messages_by_role("system")),
            "user_messages": len(self.get_messages_by_role("user")),
            "assistant_messages": len(self.get_messages_by_role("assistant")),
            "model": self.model,
            "context_window_size": self.context_window_size,
            "context_window_threshold": self.context_window_threshold,
            "context_usage_percent": round((self.last_prompt_tokens / self.context_window_size) * 100, 1)
            if self.last_prompt_tokens > 0
            else 0.0,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
        }


def run_interactive_chat(  # noqa: PLR0915, PLR0912, C901
    base_url: str,
    auth_token: str,
    model: str,
    debug: bool = False,
    system_prompt: str | None = None,
    context_window_size: int | None = None,
) -> None:
    """
    Run an interactive chat session with command history support.

    Args:
        base_url: Base URL of the deployed API
        auth_token: Authentication token
        model: Model name to use
        debug: Show debug and info log messages if True
        system_prompt: Optional system prompt for conversation context
        context_window_size: Optional context window size (defaults to 2048 if not provided)
    """
    # Configure logging level based on debug flag
    if not debug:
        # Suppress INFO and DEBUG messages during chat
        logging.getLogger("merle").setLevel(logging.WARNING)
        # Also suppress httpx logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
    else:
        # Keep INFO level for debug mode
        logging.getLogger("merle").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Use provided context window size, or fall back to default
    if context_window_size is None:
        context_window_size = 2048
        logger.info("No context window size provided, using default: 2048 tokens")

    # Set up command history with readline
    history_file = Path.home() / ".merle_history"
    try:
        # Load existing history if available
        if history_file.exists():
            readline.read_history_file(str(history_file))
        # Set maximum history size
        readline.set_history_length(1000)
    except (OSError, PermissionError) as e:
        # Continue without history if there's an issue
        logger.warning(f"Could not load command history: {e}")

    client = ChatClient(
        base_url, auth_token, model, system_prompt=system_prompt, context_window_size=context_window_size
    )

    # Display header with model information
    print("=" * 70)
    print("  MERLE CHAT SESSION")
    print(f"  Model: {model}")
    print(f"  URL: {base_url}")
    print("=" * 70)
    print("\nCommands:")
    print("  /reset  - Clear conversation history")
    print("  /status - Show conversation statistics")
    print("  /exit or /quit - Exit chat")
    print("=" * 70)

    try:
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if not user_input:
                    # Remove empty input from history
                    history_len = readline.get_current_history_length()
                    if history_len > 0:
                        readline.remove_history_item(history_len - 1)
                    continue

                # Handle commands - remove them from history since they're not user messages
                if user_input.startswith("/"):
                    # Remove command from history
                    history_len = readline.get_current_history_length()
                    if history_len > 0:
                        readline.remove_history_item(history_len - 1)

                    if user_input.lower() in ["/exit", "/quit"]:
                        print("Goodbye!")
                        break

                    if user_input.lower() == "/reset":
                        client.reset()
                        print(f"Conversation history cleared. Still connected to: {model}")
                        continue

                    if user_input.lower() == "/status":
                        summary = client.get_conversation_summary()
                        print("\nConversation Status:")
                        print(f"  Model: {summary['model']}")
                        print(f"  Total messages: {summary['total_messages']}")
                        print(f"  System messages: {summary['system_messages']}")
                        print(f"  User messages: {summary['user_messages']}")
                        print(f"  Assistant messages: {summary['assistant_messages']}")
                        print("\nContext Window:")
                        print(f"  Size: {summary['context_window_size']} tokens")
                        print(f"  Threshold (auto-trim): {summary['context_window_threshold']} tokens")
                        print(f"  Current usage: {summary['context_usage_percent']}%")
                        print("\nToken Usage:")
                        print(f"  Total tokens (session): {summary['total_tokens']}")
                        print(f"  Total prompt tokens: {summary['total_prompt_tokens']}")
                        print(f"  Total completion tokens: {summary['total_completion_tokens']}")
                        print(
                            f"  Last exchange - prompt: {summary['last_prompt_tokens']}, "
                            f"completion: {summary['last_completion_tokens']}"
                        )
                        continue

                # Send message and get response
                client.chat(user_input, prompt=f"[{model}]: ")
                print()  # Newline after response

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except httpx.HTTPError as e:
                print(f"\n\nError: {e}", file=sys.stderr)
                print("Please check your connection and try again.")
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                print(f"\n\nUnexpected error: {e}", file=sys.stderr)
                break
    finally:
        # Save command history on exit
        try:
            readline.write_history_file(str(history_file))
            logger.debug(f"Saved command history to {history_file}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not save command history: {e}")
