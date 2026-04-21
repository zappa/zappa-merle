"""Tests for the generated API Gateway authorizer."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def authorizer_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load merle/templates/authorizer.py as a module with a known API_KEY."""
    monkeypatch.setenv("API_KEY", "secret-token")
    path = Path(__file__).parent.parent / "merle" / "templates" / "authorizer.py"
    spec = importlib.util.spec_from_file_location("merle_authorizer_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METHOD_ARN = "arn:aws:execute-api:ap-northeast-1:123456789012:abcdef1234/dev/GET/"
API_WIDE = "arn:aws:execute-api:ap-northeast-1:123456789012:abcdef1234/*/*/*"


class TestApiWideResource:
    """_api_wide_resource strips the stage/method/path suffix and adds the wildcard."""

    def test_rewrites_single_method_arn_to_api_wildcard(self, authorizer_module):
        assert authorizer_module._api_wide_resource(METHOD_ARN) == API_WIDE

    def test_rewrites_post_arn_to_same_wildcard(self, authorizer_module):
        post_arn = "arn:aws:execute-api:ap-northeast-1:123456789012:abcdef1234/dev/POST/api/generate"
        assert authorizer_module._api_wide_resource(post_arn) == API_WIDE

    def test_empty_input_passes_through(self, authorizer_module):
        assert authorizer_module._api_wide_resource("") == ""


class TestLambdaHandler:
    """Cached authorizer responses must permit the whole API for the token, not one method."""

    def test_allow_policy_uses_wildcard_resource(self, authorizer_module):
        response = authorizer_module.lambda_handler({"authorizationToken": "secret-token", "methodArn": METHOD_ARN}, {})
        statement = response["policyDocument"]["Statement"][0]
        assert statement["Effect"] == "Allow"
        assert statement["Resource"] == API_WIDE

    def test_deny_policy_uses_wildcard_resource(self, authorizer_module):
        response = authorizer_module.lambda_handler({"authorizationToken": "wrong-token", "methodArn": METHOD_ARN}, {})
        statement = response["policyDocument"]["Statement"][0]
        assert statement["Effect"] == "Deny"
        assert statement["Resource"] == API_WIDE
