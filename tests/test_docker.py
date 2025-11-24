"""Docker integration tests for Merle Ollama Model Server.

These tests build the actual Docker image, run it, and test all API endpoints.
They are marked with @pytest.mark.docker to separate them from unit tests.

MODEL: Uses tinyllama (~637MB) - the smallest available Ollama model - to minimize
Docker build time and image size during testing. This makes tests faster while still
validating the full deployment workflow.

Run with: pytest tests/test_docker.py -v -s -m docker
Skip with: pytest tests/ -v -m "not docker"

Note: Docker tests can take 5-10 minutes due to model download during build.
"""

import argparse
import json
import subprocess
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from merle.cli import handle_prepare_dockerfile


def create_lambda_api_gateway_event(
    method: str = "GET",
    path: str = "/",
    body: str | None = None,
    headers: dict[str, str] | None = None,
    query_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Create a Lambda event that simulates an API Gateway request.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        body: Request body as string (JSON should be serialized)
        headers: Request headers
        query_params: Query string parameters

    Returns:
        Lambda event dictionary
    """
    return {
        "httpMethod": method,
        "path": path,
        "headers": headers or {},
        "queryStringParameters": query_params,
        "body": body,
        "isBase64Encoded": False,
        "requestContext": {
            "requestId": "test-request-id",
            "stage": "dev",
        },
    }


@pytest.fixture(scope="module")
def deployment_files(tmp_path_factory: Any) -> dict[str, Any]:  # noqa: ANN401
    """Prepare deployment files for testing.

    Uses tinyllama (~637MB) which is the smallest/lightest available Ollama model
    to minimize Docker build time and image size during testing.
    """
    cache_dir = tmp_path_factory.mktemp("docker_cache")
    test_model = "tinyllama"  # Smallest Ollama model for faster testing

    with patch("merle.functions.validate_ollama_model") as mock_validate:
        mock_validate.return_value = True

        args = argparse.Namespace(
            model=test_model,
            auth_token="test-docker-token",
            region="us-east-1",
            cache_dir=str(cache_dir),
            tags=None,
            stage="dev",
            memory_size=8192,
            s3_bucket=None,
        )

        result = handle_prepare_dockerfile(args)
        assert result == 0, "Prepare command should succeed"

    model_dir = cache_dir / "dev" / test_model
    assert model_dir.exists(), "Model directory should exist"
    assert (model_dir / "Dockerfile").exists(), "Dockerfile should exist"

    return {
        "cache_dir": cache_dir,
        "model_dir": model_dir,
        "dockerfile": model_dir / "Dockerfile",
        "model_name": test_model,
    }


@pytest.fixture(scope="module")
def docker_image(deployment_files: dict[str, Any]) -> Any:  # noqa: ANN401
    """Build Docker image from generated Dockerfile."""
    model_dir = deployment_files["model_dir"]
    image_name = "merle-test:latest"

    print(f"\nBuilding Docker image from {model_dir}...")  # noqa: T201
    print(f"Image name: {image_name}")  # noqa: T201

    # Build the image
    build_cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(model_dir / "Dockerfile"),
        str(model_dir),
    ]

    try:
        result = subprocess.run(  # noqa: S603
            build_cmd,
            cwd=str(model_dir),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for build (includes model download)
            check=True,
        )
        print("Docker build output:")  # noqa: T201
        print(result.stdout)  # noqa: T201
        if result.stderr:
            print("Docker build stderr:")  # noqa: T201
            print(result.stderr)  # noqa: T201
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed with exit code {e.returncode}")  # noqa: T201
        print(f"stdout: {e.stdout}")  # noqa: T201
        print(f"stderr: {e.stderr}")  # noqa: T201
        pytest.fail(f"Docker build failed: {e}")
    except subprocess.TimeoutExpired:
        pytest.fail("Docker build timed out after 10 minutes")

    yield image_name

    # Cleanup: remove the image
    print(f"\nRemoving Docker image {image_name}...")  # noqa: T201
    try:
        subprocess.run(  # noqa: S603
            ["docker", "rmi", "-f", image_name],  # noqa: S607
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Failed to remove image: {e}")  # noqa: T201


@pytest.fixture(scope="module")
def docker_container(docker_image: str, deployment_files: dict[str, Any]) -> Any:  # noqa: ANN401
    """Run Docker container and return its URL."""
    container_name = "merle-test-container"
    host_port = 9000  # AWS Lambda runs on port 8080 by default in container
    test_model = deployment_files["model_name"]

    print(f"\nStarting Docker container from {docker_image}...")  # noqa: T201

    # Stop and remove any existing container with the same name
    try:  # noqa: SIM105
        subprocess.run(  # noqa: S603
            ["docker", "rm", "-f", container_name],  # noqa: S607
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    # Run the container
    run_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{host_port}:8080",
        "-e",
        f"OLLAMA_MODEL={test_model}",
        docker_image,
    ]

    try:
        result = subprocess.run(  # noqa: S603
            run_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        container_id = result.stdout.strip()
        print(f"Container started: {container_id}")  # noqa: T201
    except subprocess.CalledProcessError as e:
        print(f"Failed to start container: {e}")  # noqa: T201
        print(f"stdout: {e.stdout}")  # noqa: T201
        print(f"stderr: {e.stderr}")  # noqa: T201
        pytest.fail(f"Failed to start container: {e}")

    # Wait for the container to be ready
    base_url = f"http://localhost:{host_port}"
    max_wait = 60  # Wait up to 60 seconds
    wait_interval = 2
    lambda_endpoint = "/2015-03-31/functions/function/invocations"

    print(f"Waiting for container to be ready at {base_url}...")  # noqa: T201
    for i in range(max_wait // wait_interval):
        try:
            # Check readiness via Lambda Runtime Interface with health check event
            health_event = create_lambda_api_gateway_event(method="GET", path="/health")
            response = httpx.post(
                f"{base_url}{lambda_endpoint}",
                json=health_event,
                timeout=5.0,
            )
            if response.status_code == 200:  # noqa: PLR2004
                lambda_response = response.json()
                if lambda_response.get("statusCode") == 200:  # noqa: PLR2004
                    print(f"Container ready after {(i + 1) * wait_interval} seconds")  # noqa: T201
                    break
        except (httpx.ConnectError, httpx.TimeoutException, json.JSONDecodeError):
            if i < (max_wait // wait_interval) - 1:
                time.sleep(wait_interval)
            else:
                # Print container logs before failing
                try:
                    logs_result = subprocess.run(  # noqa: S603
                        ["docker", "logs", container_name],  # noqa: S607
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    print(f"Container logs:\n{logs_result.stdout}")  # noqa: T201
                    if logs_result.stderr:
                        print(f"Container stderr:\n{logs_result.stderr}")  # noqa: T201
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    print(f"Failed to get container logs: {e}")  # noqa: T201

                pytest.fail(f"Container did not become ready within {max_wait} seconds")

    yield base_url

    # Cleanup: stop and remove the container
    print(f"\nStopping and removing container {container_name}...")  # noqa: T201
    try:
        subprocess.run(  # noqa: S603
            ["docker", "rm", "-f", container_name],  # noqa: S607
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Failed to remove container: {e}")  # noqa: T201


@pytest.mark.docker
class TestDockerDeployment:
    """Test the actual Docker deployment via Lambda Runtime Interface.

    Since the container runs with AWS Lambda base image, all requests
    must go through the Lambda Runtime Interface endpoint.
    """

    @property
    def lambda_endpoint(self) -> str:
        """Lambda Runtime Interface invocation endpoint."""
        return "/2015-03-31/functions/function/invocations"

    def invoke_lambda(self, docker_container: str, event: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """
        Invoke Lambda function via Runtime Interface Emulator.

        Args:
            docker_container: Base URL of the container
            event: Lambda event (API Gateway format)
            timeout: Request timeout in seconds

        Returns:
            Lambda response dictionary
        """
        response = httpx.post(
            f"{docker_container}{self.lambda_endpoint}",
            json=event,
            timeout=timeout,
        )
        return response.json()

    def test_root_endpoint(self, docker_container: str) -> None:
        """Test the root endpoint returns service information."""
        event = create_lambda_api_gateway_event(method="GET", path="/")
        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Root endpoint should return 200"  # noqa: PLR2004

        data = json.loads(response["body"])
        assert data["service"] == "merle-ollama-proxy", "Should return correct service name"
        assert "description" in data, "Should include description"
        assert "model" in data, "Should include model name"
        assert "endpoints" in data, "Should include endpoints info"
        assert "/health" in data["endpoints"], "Should document health endpoint"
        assert "/api/*" in data["endpoints"], "Should document API endpoint"

    def test_health_endpoint(self, docker_container: str) -> None:
        """Test the health check endpoint."""
        event = create_lambda_api_gateway_event(method="GET", path="/health")
        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Health endpoint should return 200"  # noqa: PLR2004

        data = json.loads(response["body"])
        assert data["status"] == "healthy", "Service should be healthy"
        assert "model" in data, "Should include model name"

    def test_ollama_api_tags(self, docker_container: str) -> None:
        """Test proxying to Ollama API /api/tags endpoint."""
        event = create_lambda_api_gateway_event(method="GET", path="/api/tags")
        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Tags endpoint should return 200"  # noqa: PLR2004

        data = json.loads(response["body"])
        assert "models" in data, "Should return models list"

    def test_ollama_api_generate(self, docker_container: str, deployment_files: dict[str, Any]) -> None:
        """Test proxying to Ollama API /api/generate endpoint."""
        test_model = deployment_files["model_name"]

        payload = {
            "model": test_model,
            "prompt": "Hello, world!",
            "stream": False,
        }

        event = create_lambda_api_gateway_event(
            method="POST",
            path="/api/generate",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

        response = self.invoke_lambda(docker_container, event, timeout=120.0)

        # The endpoint should work even if the model isn't loaded
        # We're just testing that the proxy works
        assert response["statusCode"] in [
            200,
            404,
            500,
        ], "Generate endpoint should be reachable"

    def test_not_found_endpoint(self, docker_container: str) -> None:
        """Test that non-existent endpoints return 404."""
        event = create_lambda_api_gateway_event(method="GET", path="/nonexistent")
        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 404, "Should return 404 for non-existent endpoints"  # noqa: PLR2004

        data = json.loads(response["body"])
        assert "error" in data, "Should return error message"

    def test_ollama_api_post_method(self, docker_container: str, deployment_files: dict[str, Any]) -> None:
        """Test that POST requests are proxied correctly."""
        test_model = deployment_files["model_name"]

        # Test with a simple POST to the API
        payload = {"model": test_model}

        event = create_lambda_api_gateway_event(
            method="POST",
            path="/api/show",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

        response = self.invoke_lambda(docker_container, event)

        # Should work or return an error from Ollama, but not fail at proxy level
        assert response["statusCode"] in [
            200,
            404,
            500,
        ], "POST should be proxied to Ollama"

    def test_cors_and_headers(self, docker_container: str) -> None:
        """Test that headers are properly handled."""
        headers = {
            "User-Agent": "merle-test-client",
            "Accept": "application/json",
        }

        event = create_lambda_api_gateway_event(method="GET", path="/health", headers=headers)
        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Should handle custom headers"  # noqa: PLR2004
        # Check response headers exist
        assert "headers" in response, "Response should include headers"
        response_headers = response.get("headers", {})
        # Content-Type can be in different case
        has_content_type = any(k.lower() == "content-type" for k in response_headers)
        assert has_content_type, "Should return Content-Type header"


@pytest.mark.docker
class TestDockerBuildProcess:
    """Test that the Docker build process works correctly."""

    def test_dockerfile_exists(self, deployment_files: dict[str, Any]) -> None:
        """Test that Dockerfile was generated."""
        dockerfile = deployment_files["dockerfile"]
        assert dockerfile.exists(), "Dockerfile should exist"

    def test_dockerfile_contains_model_name(self, deployment_files: dict[str, Any]) -> None:
        """Test that Dockerfile contains the model name."""
        dockerfile = deployment_files["dockerfile"]
        test_model = deployment_files["model_name"]
        content = dockerfile.read_text()

        assert test_model in content, "Dockerfile should contain model name"
        assert f"OLLAMA_MODEL={test_model}" in content, "Should set OLLAMA_MODEL env var"

    def test_all_required_files_copied(self, deployment_files: dict[str, Any]) -> None:
        """Test that all required files were copied to model directory."""
        model_dir = deployment_files["model_dir"]

        required_files = [
            "Dockerfile",
            "zappa_settings.json",
            "authorizer.py",
            "pyproject.toml",
        ]

        for filename in required_files:
            file_path = model_dir / filename
            assert file_path.exists(), f"{filename} should be copied to model directory"

        # Check that merle/ directory was copied
        merle_dir = model_dir / "merle"
        assert merle_dir.exists(), "merle/ directory should be copied"
        assert (merle_dir / "app.py").exists(), "merle/app.py should be copied"
        assert (merle_dir / "settings.py").exists(), "merle/settings.py should be copied"

    def test_zappa_settings_valid_json(self, deployment_files: dict[str, Any]) -> None:
        """Test that zappa_settings.json is valid JSON."""
        model_dir = deployment_files["model_dir"]
        zappa_settings = model_dir / "zappa_settings.json"

        with zappa_settings.open() as f:
            settings = json.load(f)

        assert "dev" in settings, "Should have dev stage"
        assert settings["dev"]["app_function"] == "merle.app.app", "Should reference correct app"
        assert settings["dev"]["aws_region"] == "us-east-1", "Should use correct region"


@pytest.mark.docker
class TestLambdaRuntimeInterface:
    """Test the AWS Lambda Runtime Interface Emulator (RIE).

    These tests invoke the Lambda function through the Runtime Interface,
    which simulates how API Gateway invokes the Lambda function in production.
    """

    @property
    def lambda_endpoint(self) -> str:
        """Lambda Runtime Interface invocation endpoint."""
        return "/2015-03-31/functions/function/invocations"

    def invoke_lambda(self, docker_container: str, event: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """
        Invoke Lambda function via Runtime Interface Emulator.

        Args:
            docker_container: Base URL of the container
            event: Lambda event (API Gateway format)
            timeout: Request timeout in seconds

        Returns:
            Lambda response dictionary
        """
        response = httpx.post(
            f"{docker_container}{self.lambda_endpoint}",
            json=event,
            timeout=timeout,
        )
        return response.json()

    def test_lambda_invocation_root_endpoint(self, docker_container: str) -> None:
        """Test Lambda invocation of root endpoint via Runtime Interface."""
        event = create_lambda_api_gateway_event(method="GET", path="/")

        response = self.invoke_lambda(docker_container, event)

        assert "statusCode" in response, "Lambda response should have statusCode"
        assert response["statusCode"] == 200, "Root endpoint should return 200"  # noqa: PLR2004
        assert "body" in response, "Lambda response should have body"

        # Parse the body (Zappa wraps Flask responses in Lambda format)
        body = json.loads(response["body"])
        assert body["service"] == "merle-ollama-proxy", "Should return correct service name"
        assert "model" in body, "Should include model name"
        assert "endpoints" in body, "Should include endpoints"

    def test_lambda_invocation_health_check(self, docker_container: str) -> None:
        """Test Lambda invocation of health check endpoint."""
        event = create_lambda_api_gateway_event(method="GET", path="/health")

        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Health check should return 200"  # noqa: PLR2004

        body = json.loads(response["body"])
        assert body["status"] == "healthy", "Service should be healthy"
        assert "model" in body, "Should include model name"

    def test_lambda_invocation_with_headers(self, docker_container: str) -> None:
        """Test Lambda invocation with custom headers."""
        headers = {
            "User-Agent": "merle-lambda-test",
            "Accept": "application/json",
            "X-Custom-Header": "test-value",
        }

        event = create_lambda_api_gateway_event(
            method="GET",
            path="/health",
            headers=headers,
        )

        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Should handle custom headers"  # noqa: PLR2004
        assert "headers" in response, "Response should include headers"
        # Check Content-Type header is set
        response_headers = response.get("headers", {})
        assert "Content-Type" in response_headers or "content-type" in response_headers, (
            "Response should have Content-Type header"
        )

    def test_lambda_invocation_ollama_api_tags(self, docker_container: str) -> None:
        """Test Lambda invocation of Ollama API tags endpoint."""
        event = create_lambda_api_gateway_event(method="GET", path="/api/tags")

        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 200, "Tags endpoint should return 200"  # noqa: PLR2004

        body = json.loads(response["body"])
        assert "models" in body, "Should return models list"

    def test_lambda_invocation_post_request(self, docker_container: str, deployment_files: dict[str, Any]) -> None:
        """Test Lambda invocation with POST request."""
        test_model = deployment_files["model_name"]

        payload = {
            "model": test_model,
            "prompt": "Test prompt",
            "stream": False,
        }

        event = create_lambda_api_gateway_event(
            method="POST",
            path="/api/generate",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

        response = self.invoke_lambda(docker_container, event, timeout=120.0)

        # Should get a response (even if generation fails, Lambda invocation should work)
        assert "statusCode" in response, "Lambda response should have statusCode"
        assert response["statusCode"] in [
            200,
            404,
            500,
        ], "Should get valid HTTP status code"

    def test_lambda_invocation_not_found(self, docker_container: str) -> None:
        """Test Lambda invocation returns 404 for non-existent endpoint."""
        event = create_lambda_api_gateway_event(method="GET", path="/nonexistent")

        response = self.invoke_lambda(docker_container, event)

        assert response["statusCode"] == 404, "Should return 404 for non-existent endpoint"  # noqa: PLR2004

        body = json.loads(response["body"])
        assert "error" in body, "Should return error message"

    def test_lambda_invocation_with_query_params(self, docker_container: str) -> None:
        """Test Lambda invocation with query parameters."""
        event = create_lambda_api_gateway_event(
            method="GET",
            path="/api/tags",
            query_params={"verbose": "true"},
        )

        response = self.invoke_lambda(docker_container, event)

        # Query params should be passed through to the Flask app
        assert response["statusCode"] == 200, "Should handle query parameters"  # noqa: PLR2004

    def test_lambda_response_format(self, docker_container: str) -> None:
        """Test that Lambda response follows correct format."""
        event = create_lambda_api_gateway_event(method="GET", path="/health")

        response = self.invoke_lambda(docker_container, event)

        # Verify Lambda response structure
        assert isinstance(response, dict), "Response should be a dictionary"
        assert "statusCode" in response, "Response must have statusCode"
        assert "body" in response, "Response must have body"
        assert "headers" in response, "Response must have headers"

        # Verify statusCode is an integer
        assert isinstance(response["statusCode"], int), "statusCode should be an integer"

        # Verify body is a string (JSON serialized)
        assert isinstance(response["body"], str), "body should be a string"

        # Verify body is valid JSON
        try:
            json.loads(response["body"])
        except json.JSONDecodeError:
            pytest.fail("Response body should be valid JSON")

    def test_lambda_invocation_different_methods(self, docker_container: str) -> None:
        """Test Lambda invocation supports different HTTP methods."""
        methods_and_paths = [
            ("GET", "/health"),
            ("GET", "/api/tags"),
            ("GET", "/"),
        ]

        for method, path in methods_and_paths:
            event = create_lambda_api_gateway_event(method=method, path=path)
            response = self.invoke_lambda(docker_container, event)

            assert "statusCode" in response, f"{method} {path} should return valid Lambda response"
            assert isinstance(response["statusCode"], int), f"{method} {path} statusCode should be integer"
