"""Chat client for interacting with deployed Ollama models."""

import json
import logging
import sys

import httpx

logger = logging.getLogger(__name__)


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

    def _make_request(self, messages: list[dict[str, str]]) -> str:
        """
        Make a streaming request to the Ollama API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

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

        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content

                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON: {line}")
                            continue

                print()  # Newline after streaming response
                return full_response

        except httpx.HTTPStatusError as e:
            # For streaming responses, we need to read the content first
            try:
                error_body = e.response.read().decode("utf-8")
            except (UnicodeDecodeError, AttributeError, OSError):
                error_body = "<unable to read response body>"
            error_msg = f"HTTP error {e.response.status_code}: {error_body}"
            logger.error(error_msg)
            raise
        except httpx.ConnectError as e:
            error_msg = f"Connection error: {e}"
            logger.error(error_msg)
            raise
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {e}"
            logger.error(error_msg)
            raise

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: User's message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Get response
        response = self._make_request(self.conversation_history)

        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")


def run_interactive_chat(base_url: str, auth_token: str, model: str) -> None:
    """
    Run an interactive chat session.

    Args:
        base_url: Base URL of the deployed API
        auth_token: Authentication token
        model: Model name to use
    """
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
            user_input = input("\nYou: ").strip()

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
            print(f"\n[{model}]: ", end="", flush=True)
            client.chat(user_input)

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
