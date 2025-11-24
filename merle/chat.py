"""Chat client for interacting with deployed Ollama models."""

import json
import logging
import sys
import threading
import time

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

    def __init__(self, base_url: str, auth_token: str, model: str) -> None:
        """
        Initialize the chat client.

        Args:
            base_url: Base URL of the deployed API (e.g., https://xxx.execute-api.us-east-1.amazonaws.com)
            auth_token: Authentication token for API access
            model: Model name to use for chat
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.model = model
        self.conversation_history: list[dict[str, str]] = []

    def _process_stream_line(
        self, line: str, spinner: WaitingCursor, first_content: bool, prompt: str | None = None
    ) -> tuple[str, bool]:
        """
        Process a single line from the streaming response.

        Args:
            line: JSON line from the stream
            spinner: WaitingCursor instance to stop on first content
            first_content: Whether this is the first content received
            prompt: Optional prompt to display before first content (e.g., "[model]: ")

        Returns:
            Tuple of (content, is_first_content)
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
                    return content, first_content

            if data.get("done", False):
                return "", first_content
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON: {line}")

        return "", first_content

    def _make_request(self, messages: list[dict[str, str]], prompt: str | None = None) -> str:
        """
        Make a streaming request to the Ollama API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            prompt: Optional prompt to display before first content (e.g., "[model]: ")

        Returns:
            Complete response text

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

                        content, first_content = self._process_stream_line(line, spinner, first_content, prompt)
                        full_response += content

                        # Check if done
                        try:
                            if json.loads(line).get("done", False):
                                break
                        except json.JSONDecodeError:
                            pass

                spinner.stop()
                return full_response

        except httpx.HTTPStatusError as e:
            spinner.stop()
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
            return error.response.read().decode("utf-8")
        except (UnicodeDecodeError, AttributeError, OSError):
            return "<unable to read response body>"

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
        self.conversation_history.append({"role": "user", "content": user_message})

        # Get response
        response = self._make_request(self.conversation_history, prompt=prompt)

        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")


def run_interactive_chat(base_url: str, auth_token: str, model: str, debug: bool = False) -> None:
    """
    Run an interactive chat session.

    Args:
        base_url: Base URL of the deployed API
        auth_token: Authentication token
        model: Model name to use
        debug: Show debug and info log messages if True
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

    client = ChatClient(base_url, auth_token, model)

    # Display header with model information
    print("=" * 70)
    print("  MERLE CHAT SESSION")
    print(f"  Model: {model}")
    print(f"  URL: {base_url}")
    print("=" * 70)
    print("\nCommands:")
    print("  /reset - Clear conversation history")
    print("  /exit or /quit - Exit chat")
    print("=" * 70)

    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/exit", "/quit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "/reset":
                client.reset()
                print(f"Conversation history cleared. Still connected to: {model}")
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
