"""Deployment management for Merle Ollama Model Server.

This module provides the DeploymentManager class for handling infrastructure
preparation, Docker image building, ECR operations, and Zappa deployment.
"""

import base64
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import boto3
from zappa.cli import ZappaCLI
from zappa.core import Zappa

from merle.functions import (
    generate_from_template,
    get_model_cache_dir,
    get_model_context_window_size,
    normalize_model_name,
    sanitize_for_cloudformation,
    update_model_config,
    validate_ollama_model,
)
from merle.settings import REGION

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages infrastructure preparation and deployment for Ollama models on AWS Lambda.

    This class encapsulates all deployment-related operations including:
    - Preparing deployment files (Dockerfile, zappa_settings.json)
    - Building and pushing Docker images to ECR
    - Deploying with Zappa
    - Tearing down deployments

    Args:
        model_name: Ollama model name (e.g., 'llama2', 'mistral')
        cache_dir: Base cache directory for deployment files
        project_name: Project name for AWS resource naming
        stage: Deployment stage (default: 'dev')
        region: AWS region (defaults to settings.REGION)
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        project_name: str,
        stage: str = "dev",
        region: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.project_name = project_name
        self.stage = stage
        self.region = region or REGION
        self._model_cache_dir: Path | None = None
        self._ecr_image_uri: str | None = None

    @property
    def model_cache_dir(self) -> Path:
        """Get the model-stage-specific cache directory."""
        if self._model_cache_dir is None:
            self._model_cache_dir = get_model_cache_dir(self.cache_dir, self.model_name, self.stage)
        return self._model_cache_dir

    @property
    def zappa_settings_path(self) -> Path:
        """Get path to zappa_settings.json."""
        return self.model_cache_dir / "zappa_settings.json"

    @property
    def is_prepared(self) -> bool:
        """Check if deployment files have been prepared."""
        return self.zappa_settings_path.exists()

    @property
    def normalized_model_name(self) -> str:
        """Get normalized model name for ECR/AWS resource naming."""
        return normalize_model_name(self.model_name)

    @property
    def ecr_repo_name(self) -> str:
        """Get ECR repository name."""
        return f"merle-{self.normalized_model_name}"

    def prepare(
        self,
        auth_token: str,
        s3_bucket: str,
        tags: dict[str, str] | None = None,
        memory_size: int = 8192,
        system_prompt: str | None = None,
        skip_model_download: bool = False,
    ) -> Path:
        """
        Prepare all necessary files for deployment.

        This method handles model size detection and splitting automatically:
        - Downloads the model using local Ollama
        - Calculates if the model fits in a Docker image (~8GB limit)
        - If too large, splits the model and uploads overflow to S3
        - Generates appropriate Dockerfile (standard or split mode)

        Args:
            auth_token: Authentication token for API access
            s3_bucket: S3 bucket name for Zappa deployment
            tags: Optional AWS resource tags
            memory_size: Lambda function memory size in MB (default: 8192)
            system_prompt: Optional system prompt for chat context
            skip_model_download: Skip model download (for testing)

        Returns:
            Path to the model-stage-specific cache directory

        Raises:
            ValueError: If validation fails or required parameters are missing
        """
        logger.info(f"Preparing deployment files for model: {self.model_name}, stage: {self.stage}")

        # Validate model name
        validate_ollama_model(self.model_name)

        # Create cache directory
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Get template directory from installed package
        template_dir = Path(__file__).parent / "templates"

        # Get consuming project root from current working directory
        consuming_project_root = Path.cwd()

        # Prepare replacements
        tags_dict = tags or {}

        replacements = {
            "OLLAMA_MODEL": self.model_name,
            "AWS_REGION": self.region,
            "AUTH_TOKEN": auth_token,
            "TAGS_JSON": json.dumps(tags_dict),
        }

        # Determine if we need to download and potentially split the model
        use_split = False
        split_metadata = None

        if not skip_model_download:
            use_split, split_metadata = self._handle_model_download(s3_bucket)

        # Add split-specific replacements for Dockerfile template
        if use_split:
            replacements["OLLAMA_MODELS_PATH"] = "/tmp/models"  # noqa: S108
            replacements["SPLIT_MODEL_ENV"] = "    MERLE_SPLIT_MODEL=true \\\n"
            replacements["SPLIT_METADATA_CHECK"] = (
                ' && \\\n    cat /var/task/models/split_metadata.json 2>/dev/null || echo "No split metadata"'
            )
        else:
            replacements["OLLAMA_MODELS_PATH"] = "/var/task/models"
            replacements["SPLIT_MODEL_ENV"] = ""
            replacements["SPLIT_METADATA_CHECK"] = ""

        # Generate Dockerfile from template
        generate_from_template(
            template_path=template_dir / "Dockerfile.template",
            output_path=self.model_cache_dir / "Dockerfile",
            replacements=replacements,
        )

        # Copy authorizer.py
        authorizer_src = template_dir / "authorizer.py"
        if authorizer_src.exists():
            shutil.copy2(authorizer_src, self.model_cache_dir / "authorizer.py")
            logger.info("Copied authorizer.py to model cache directory")

        # Get context window size for the model
        context_window_size = get_model_context_window_size(self.model_name)

        # Calculate ephemeral storage needed based on model size
        model_size_gb = split_metadata.get("total_size_gb", 0) if split_metadata else 0
        ephemeral_storage = self._calculate_ephemeral_storage(model_size_gb)

        # Generate zappa_settings.json
        self._generate_zappa_settings(
            auth_token=auth_token,
            s3_bucket=s3_bucket,
            tags=tags_dict,
            memory_size=memory_size,
            context_window_size=context_window_size,
            use_split=use_split,
            ephemeral_storage=ephemeral_storage,
        )

        # Copy pyproject.toml from consuming project
        pyproject_src = consuming_project_root / "pyproject.toml"
        if pyproject_src.exists():
            shutil.copy2(pyproject_src, self.model_cache_dir / "pyproject.toml")
            logger.info(f"Copied pyproject.toml from consuming project: {consuming_project_root}")
        else:
            logger.warning(f"pyproject.toml not found in consuming project: {consuming_project_root}")

        # Update configuration
        update_model_config(
            cache_dir=self.cache_dir,
            model_name=self.model_name,
            auth_token=auth_token,
            region=self.region,
            tags=tags_dict if tags_dict else None,
            stage=self.stage,
            system_prompt=system_prompt,
            context_window_size=context_window_size,
            use_split=use_split,
            split_config=split_metadata,
        )

        logger.info(f"Successfully prepared deployment files in {self.model_cache_dir}")
        return self.model_cache_dir

    def _handle_model_download(self, s3_bucket: str) -> tuple[bool, dict | None]:
        """
        Handle model download and potential splitting.

        Args:
            s3_bucket: S3 bucket for overflow storage

        Returns:
            Tuple of (use_split, split_metadata)
        """
        from merle.model_split import (  # noqa: PLC0415
            calculate_model_size,
            copy_model_to_output,
            download_model_locally,
            needs_splitting,
            prepare_split_model,
        )

        logger.info(f"Downloading model {self.model_name} using local Ollama...")
        try:
            download_model_locally(self.model_name)
        except RuntimeError as e:
            logger.exception("Failed to download model")
            raise ValueError(f"Failed to download model: {e}") from e

        # Calculate model size
        total_size, size_details = calculate_model_size(self.model_name)
        logger.info(f"Model size: {size_details['total_size_gb']} GB")

        # Check if splitting is needed
        if needs_splitting(total_size):
            logger.info("Model exceeds Docker image limit, preparing split deployment...")

            split_metadata = prepare_split_model(
                model_name=self.model_name,
                output_dir=self.model_cache_dir,
                s3_bucket=s3_bucket,
                region=self.region,
            )

            logger.info("Split model prepared:")
            logger.info(f"  - Image portion: {split_metadata['image_portion_bytes'] / (1024**3):.2f} GB")
            logger.info(f"  - S3 portion: {split_metadata['s3_portion_bytes'] / (1024**3):.2f} GB")
            logger.info(f"  - S3 URI: {split_metadata['s3']['uri']}")
            return True, split_metadata

        logger.info("Model fits in Docker image, using standard deployment")
        copy_model_to_output(self.model_name, self.model_cache_dir)
        # Return size info for ephemeral storage calculation
        return False, {"total_size_gb": size_details["total_size_gb"]}

    def _calculate_ephemeral_storage(self, model_size_gb: float) -> int:
        """
        Calculate ephemeral storage needed for Lambda based on model size.

        Lambda ephemeral storage (/tmp) ranges from 512 MB to 10,240 MB.
        We need enough space for the model files at runtime.

        Args:
            model_size_gb: Model size in GB

        Returns:
            Ephemeral storage in MB (512-10240)
        """
        # Calculate needed storage: model size + 20% buffer, rounded up to nearest 512 MB
        needed_mb = int((model_size_gb * 1024 * 1.2 + 511) // 512 * 512)

        # Clamp to Lambda limits: min 512 MB, max 10,240 MB
        ephemeral_storage = min(max(needed_mb, 512), 10240)

        logger.info(f"Setting ephemeral storage to {ephemeral_storage} MB for {model_size_gb:.2f} GB model")
        return ephemeral_storage

    def _generate_zappa_settings(
        self,
        auth_token: str,
        s3_bucket: str,
        tags: dict[str, str],
        memory_size: int = 8192,
        context_window_size: int | None = None,
        use_split: bool = False,
        ephemeral_storage: int = 5120,
    ) -> None:
        """Generate zappa_settings.json using Zappa's Python API."""
        mode_str = " (split mode)" if use_split else ""
        logger.info(f"Generating zappa_settings.json for stage '{self.stage}'{mode_str}")

        cli = ZappaCLI()
        settings_dict = cli._generate_settings_dict(stage=self.stage, config_args=[])

        sanitized_project = sanitize_for_cloudformation(self.project_name)
        stage_config = settings_dict[self.stage]

        # Configure embedded authorizer
        authorizer_config = {
            "function": "authorizer.lambda_handler",
            "token_header": "X-API-Key",
            "result_ttl": 300,
        }

        # Build environment variables
        models_path = "/tmp/models"  # noqa: S108
        env_vars = {
            "OLLAMA_MODEL": self.model_name,
            "OLLAMA_URL": "http://localhost:11434",
            "OLLAMA_MODELS": models_path,
            "OLLAMA_STARTUP_TIMEOUT": "120",
            "ZAPPA_RUNNING_IN_DOCKER": "True",
            "API_KEY": auth_token,
        }

        if context_window_size:
            env_vars["OLLAMA_MODEL_CONTEXT_WINDOW_SIZE"] = str(context_window_size)

        if use_split:
            env_vars["MERLE_SPLIT_MODEL"] = "true"
            env_vars["OLLAMA_STARTUP_TIMEOUT"] = "300"

        stage_config.update(
            {
                "app_function": "merle.app.app",
                "project_name": sanitized_project,
                "s3_bucket": s3_bucket,
                "aws_region": self.region,
                "memory_size": memory_size,
                "timeout_seconds": 900,
                "environment_variables": env_vars,
                "ephemeral_storage": {"Size": ephemeral_storage},
                "keep_warm": False,
                "keep_warm_expression": "rate(4 minutes)",
                "authorizer": authorizer_config,
                "cors": True,
                "cors_allow_headers": ["Content-Type", "X-API-Key"],
                "binary_support": False,
                "tags": tags,
            }
        )

        if use_split:
            extra_permissions = [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:HeadObject"],
                    "Resource": f"arn:aws:s3:::{s3_bucket}/merle-models/*",
                }
            ]
            stage_config["extra_permissions"] = extra_permissions
            logger.info("Added S3 permissions for split model download")

        # Write to file
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        with self.zappa_settings_path.open("w") as f:
            json.dump(settings_dict, f, indent=4, sort_keys=True)

        logger.info(f"Successfully generated {self.zappa_settings_path}")

    def build_and_push_docker_image(self) -> str:
        """
        Build Docker image and push to ECR.

        Returns:
            ECR image URI

        Raises:
            RuntimeError: If build or push fails
        """
        if not self.is_prepared:
            raise RuntimeError(
                f"Deployment not prepared. Run prepare() first or use 'merle prepare --model {self.model_name}'"
            )

        ecr_client = boto3.client("ecr", region_name=self.region)

        # Create ECR repository
        try:
            logger.info(f"Creating ECR repository: {self.ecr_repo_name}")
            ecr_client.create_repository(
                repositoryName=self.ecr_repo_name,
                imageScanningConfiguration={"scanOnPush": True},
            )
            logger.info(f"Created ECR repository: {self.ecr_repo_name}")
        except ecr_client.exceptions.RepositoryAlreadyExistsException:
            logger.info(f"ECR repository already exists: {self.ecr_repo_name}")

        # Get repository URI
        response = ecr_client.describe_repositories(repositoryNames=[self.ecr_repo_name])
        repo_uri = response["repositories"][0]["repositoryUri"]
        image_uri = f"{repo_uri}:latest"
        logger.info(f"ECR image URI: {image_uri}")

        # Authenticate Docker to ECR
        self._authenticate_ecr(ecr_client)

        # Build Docker image
        self._build_docker_image(image_uri)

        # Push Docker image
        self._push_docker_image(image_uri)

        # Update zappa_settings.json with docker_image_uri
        self._update_zappa_settings_with_image_uri(image_uri)

        self._ecr_image_uri = image_uri
        return image_uri

    def _authenticate_ecr(self, ecr_client: Any) -> None:  # noqa: ANN401
        """Authenticate Docker to ECR."""
        logger.info("Authenticating with ECR...")
        auth_response = ecr_client.get_authorization_token()
        auth_token = auth_response["authorizationData"][0]["authorizationToken"]
        ecr_endpoint = auth_response["authorizationData"][0]["proxyEndpoint"]

        username, password = base64.b64decode(auth_token).decode().split(":")

        docker_login_cmd = ["docker", "login", "--username", username, "--password-stdin", ecr_endpoint]
        subprocess.run(  # noqa: S603
            docker_login_cmd,
            input=password.encode(),
            check=True,
            capture_output=True,
        )
        logger.info("Authenticated with ECR")

    def _build_docker_image(self, image_uri: str) -> None:
        """Build Docker image."""
        logger.info(f"Building Docker image: {image_uri}")
        docker_build_cmd = ["docker", "build", "-t", image_uri, "."]
        subprocess.run(  # noqa: S603
            docker_build_cmd,
            cwd=self.model_cache_dir,
            check=True,
            capture_output=False,
        )
        logger.info(f"Built Docker image: {image_uri}")

    def _push_docker_image(self, image_uri: str) -> None:
        """Push Docker image to ECR."""
        logger.info(f"Pushing Docker image to ECR: {image_uri}")
        docker_push_cmd = ["docker", "push", image_uri]
        subprocess.run(  # noqa: S603
            docker_push_cmd,
            cwd=self.model_cache_dir,
            check=True,
            capture_output=False,
        )
        logger.info(f"Pushed Docker image: {image_uri}")

    def _update_zappa_settings_with_image_uri(self, image_uri: str) -> None:
        """Update zappa_settings.json with docker_image_uri."""
        with self.zappa_settings_path.open() as f:
            settings_data = json.load(f)

        settings_data[self.stage]["docker_image_uri"] = image_uri

        with self.zappa_settings_path.open("w") as f:
            json.dump(settings_data, f, indent=4, sort_keys=True)

        logger.info(f"Updated zappa_settings.json with docker_image_uri: {image_uri}")

    def deploy(self, auth_token: str, max_retries: int = 3, retry_delay: int = 15) -> str | None:
        """
        Deploy to AWS Lambda using Zappa.

        Args:
            auth_token: Authentication token for API access
            max_retries: Maximum number of retry attempts for IAM role propagation
            retry_delay: Delay in seconds between retries

        Returns:
            Deployment URL if successful, None otherwise

        Raises:
            RuntimeError: If deployment fails
        """
        if not self.is_prepared:
            raise RuntimeError(
                f"Deployment not prepared. Run prepare() first or use 'merle prepare --model {self.model_name}'"
            )

        # Get or build image URI
        image_uri = self._ecr_image_uri
        if not image_uri:
            with self.zappa_settings_path.open() as f:
                settings_data = json.load(f)
            image_uri = settings_data.get(self.stage, {}).get("docker_image_uri")

        if not image_uri:
            raise RuntimeError("Docker image URI not found. Run build_and_push_docker_image() first.")

        # Set environment for deployment
        env = os.environ.copy()
        env["API_KEY"] = auth_token

        # Run zappa deploy with retry logic
        cmd = ["zappa", "deploy", self.stage, "--docker-image-uri", image_uri]

        for attempt in range(1, max_retries + 1):
            logger.info(f"Running (attempt {attempt}/{max_retries}): {' '.join(cmd)}")

            result = subprocess.run(  # noqa: S603
                cmd,
                env=env,
                cwd=self.model_cache_dir,
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                if result.stdout:
                    logger.info(result.stdout)
                break

            error_output = result.stderr or result.stdout or ""
            is_role_propagation_error = "role defined for the function cannot be assumed by Lambda" in error_output

            if is_role_propagation_error and attempt < max_retries:
                logger.warning(f"IAM role propagation delay detected, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue

            if result.stdout:
                logger.error(result.stdout)
            if result.stderr:
                logger.error(result.stderr)
            break

        if result.returncode != 0:
            raise RuntimeError(f"Deployment failed with exit code: {result.returncode}")

        # Get deployment URL
        deployment_url = self.get_deployment_url()
        if deployment_url:
            update_model_config(
                cache_dir=self.cache_dir,
                model_name=self.model_name,
                url=deployment_url,
                stage=self.stage,
            )
            logger.info(f"Saved deployment URL to config: {deployment_url}")

        return deployment_url

    def get_deployment_url(self) -> str | None:  # noqa: PLR0911
        """Get the deployment URL for the model."""
        if not self.is_prepared:
            return None

        try:
            with self.zappa_settings_path.open() as f:
                settings = json.load(f)

            stage_config = settings.get(self.stage, {})
            if not stage_config:
                return None

            project_name = stage_config.get("project_name")
            if not project_name:
                return None

            lambda_name = f"{project_name}-{self.stage}"
            aws_region = stage_config.get("aws_region", "us-east-1")

            # Check if CloudFormation stack exists
            cf_client = boto3.client("cloudformation", region_name=aws_region)
            try:
                cf_client.describe_stacks(StackName=lambda_name)
            except cf_client.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "ValidationError":
                    logger.debug(f"CloudFormation stack '{lambda_name}' does not exist")
                    return None
                raise

            # Get API URL using Zappa
            zappa = Zappa()
            return zappa.get_api_url(lambda_name, self.stage)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Could not get deployment URL: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Error getting deployment URL from Zappa: {e}")
            return None

    def destroy(self, skip_confirmation: bool = False) -> bool:
        """
        Tear down the deployment.

        Args:
            skip_confirmation: Skip the interactive confirmation prompt

        Returns:
            True if successful, False otherwise
        """
        if not self.is_prepared:
            logger.error(f"No deployment found for model: {self.model_name}, stage: {self.stage}")
            return False

        logger.info(f"Destroying deployment for model: {self.model_name}, stage: {self.stage}")

        # Run zappa undeploy
        cmd = ["zappa", "undeploy", self.stage, "--yes"]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=self.model_cache_dir,
            check=False,
            capture_output=False,
        )

        zappa_succeeded = result.returncode == 0
        if not zappa_succeeded:
            logger.warning(f"Zappa undeploy returned exit code {result.returncode}")

        # Clean up local files
        self._cleanup_local_files()

        return zappa_succeeded

    def _cleanup_local_files(self) -> None:
        """Clean up local deployment files and configuration."""
        from merle.functions import load_config, save_config  # noqa: PLC0415

        logger.info(f"Cleaning up local files for model: {self.model_name}, stage: {self.stage}")

        # Remove model-stage entry from config
        config = load_config(self.cache_dir)
        if self.model_name in config.get("models", {}) and self.stage in config["models"][self.model_name]:
            del config["models"][self.model_name][self.stage]
            if not config["models"][self.model_name]:
                del config["models"][self.model_name]
            save_config(self.cache_dir, config)
            logger.info(f"Removed {self.model_name} (stage: {self.stage}) from configuration")

        # Delete model cache directory
        if self.model_cache_dir.exists():
            shutil.rmtree(self.model_cache_dir)
            logger.info(f"Deleted model cache directory: {self.model_cache_dir}")
