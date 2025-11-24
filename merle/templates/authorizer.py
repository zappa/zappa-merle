"""Lambda authorizer for API Gateway token-based authentication."""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get the API key from environment variable
API_KEY = os.getenv("API_KEY")


def lambda_handler(event: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """
    Lambda authorizer handler for API Gateway.

    Args:
        event: API Gateway authorizer event
        _context: Lambda context (unused)

    Returns:
        IAM policy document allowing or denying access
    """
    token = event.get("authorizationToken", "")
    method_arn = event.get("methodArn", "")

    logger.info(f"Authorizer invoked for method: {method_arn}")

    # Validate token
    if token == API_KEY:
        logger.info("Token validation successful")
        return generate_policy("user", "Allow", method_arn)

    logger.warning("Token validation failed")
    return generate_policy("user", "Deny", method_arn)


def generate_policy(principal_id: str, effect: str, resource: str) -> dict[str, Any]:
    """
    Generate IAM policy document.

    Args:
        principal_id: User identifier
        effect: Allow or Deny
        resource: ARN of the API Gateway method

    Returns:
        IAM policy document
    """
    auth_response: dict[str, Any] = {
        "principalId": principal_id,
    }

    if effect and resource:
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "execute-api:Invoke",
                    "Effect": effect,
                    "Resource": resource,
                }
            ],
        }
        auth_response["policyDocument"] = policy_document

    return auth_response
