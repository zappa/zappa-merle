"""Tests for merle.cli module."""

import argparse
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from merle.cli import (
    get_config_directory,
    get_model_to_use,
    handle_chat,
    handle_deploy,
    handle_destroy,
    handle_list,
    handle_prepare_dockerfile,
)


class TestGetConfigDirectory:
    """Tests for get_config_directory function."""

    @patch("merle.cli.Path.home")
    def test_get_config_directory(self, mock_home: MagicMock, tmp_path: Path):
        """Test getting config directory creates ~/.merle."""
        mock_home.return_value = tmp_path

        config_dir = get_config_directory()

        assert config_dir == tmp_path / ".merle"
        assert config_dir.exists()


class TestGetModelToUse:
    """Tests for get_model_to_use function."""

    def test_get_model_specified(self, temp_cache_dir: Path):
        """Test that specified model is returned."""
        result = get_model_to_use(temp_cache_dir, "llama2")
        assert result == "llama2"

    def test_get_model_no_models_configured(self, temp_cache_dir: Path):
        """Test error when no models configured and none specified."""
        with pytest.raises(ValueError, match="No models configured"):
            get_model_to_use(temp_cache_dir, None)

    def test_get_model_single_model_auto_detect(self, temp_cache_dir: Path, sample_config: dict):
        """Test auto-detection when only one model exists."""
        # Create config with only one model
        single_model_config = {"models": {"llama2": sample_config["models"]["llama2"]}}
        config_path = temp_cache_dir / "config.json"
        config_path.write_text(json.dumps(single_model_config))

        result = get_model_to_use(temp_cache_dir, None)
        assert result == "llama2"

    def test_get_model_multiple_models_no_specification(self, temp_cache_dir: Path, sample_config: dict):
        """Test error when multiple models exist and none specified."""
        config_path = temp_cache_dir / "config.json"
        config_path.write_text(json.dumps(sample_config))

        with pytest.raises(ValueError, match="Multiple models are deployed \\(2 models"):
            get_model_to_use(temp_cache_dir, None)


class TestHandlePrepareDockerfile:
    """Tests for handle_prepare_dockerfile command."""

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.generate_unique_bucket_name")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.get_default_project_name")
    def test_prepare_success(
        self,
        mock_get_default_project: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_gen_bucket: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test successful prepare command."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_cache.return_value = temp_cache_dir / "llama2"
        mock_gen_bucket.return_value = "zappa-merle-12345678"
        mock_prepare.return_value = temp_cache_dir / "llama2"
        mock_get_default_project.return_value = "testproject"

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket=None,
            stage="dev",
            memory_size=8192,
            project=None,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 0
        mock_prepare.assert_called_once_with(
            model_name="llama2",
            cache_dir=temp_cache_dir / "testproject",
            project_name="testproject",
            auth_token="test-token",
            aws_region="us-east-1",
            tags=None,
            s3_bucket="zappa-merle-12345678",
            stage="dev",
            memory_size=8192,
            system_prompt=None,
        )

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.generate_unique_bucket_name")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.parse_tags")
    @patch("merle.cli.get_default_project_name")
    def test_prepare_with_tags(
        self,
        mock_get_default_project: MagicMock,
        mock_parse_tags: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_gen_bucket: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
        sample_tags: dict,
    ):
        """Test prepare command with tags."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_cache.return_value = temp_cache_dir / "llama2"
        mock_gen_bucket.return_value = "zappa-merle-87654321"
        mock_prepare.return_value = temp_cache_dir / "llama2"
        mock_parse_tags.return_value = sample_tags
        mock_get_default_project.return_value = "testproject"

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags="Environment=dev,Project=ollama",
            s3_bucket=None,
            stage="dev",
            memory_size=8192,
            project=None,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 0
        mock_parse_tags.assert_called_once_with("Environment=dev,Project=ollama")
        mock_prepare.assert_called_once_with(
            model_name="llama2",
            cache_dir=temp_cache_dir / "testproject",
            project_name="testproject",
            auth_token="test-token",
            aws_region="us-east-1",
            tags=sample_tags,
            s3_bucket="zappa-merle-87654321",
            stage="dev",
            memory_size=8192,
            system_prompt=None,
        )

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.parse_tags")
    def test_prepare_invalid_tags(
        self,
        mock_parse_tags: MagicMock,
        mock_get_config: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test prepare command with invalid tags format."""
        mock_get_config.return_value = temp_cache_dir
        mock_parse_tags.side_effect = ValueError("Invalid tag format")

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags="invalid-tag-format",
            stage="dev",
            memory_size=8192,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 1
        mock_prepare.assert_not_called()

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.get_config_directory")
    def test_prepare_with_custom_cache_dir(
        self,
        mock_get_config: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test prepare command with custom cache directory."""
        custom_cache = temp_cache_dir / "custom"
        mock_prepare.return_value = custom_cache / "llama2"

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=str(custom_cache),
            tags=None,
            s3_bucket=None,
            stage="dev",
            memory_size=8192,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 0
        assert custom_cache.exists()
        mock_get_config.assert_not_called()  # Should not use default config dir

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.get_default_project_name")
    def test_prepare_with_custom_s3_bucket(
        self,
        mock_get_default_project: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test prepare command with custom S3 bucket."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_cache.return_value = temp_cache_dir / "llama2"
        mock_prepare.return_value = temp_cache_dir / "llama2"
        mock_get_default_project.return_value = "testproject"

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket="my-custom-bucket",
            stage="dev",
            memory_size=8192,
            project=None,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 0
        mock_prepare.assert_called_once_with(
            model_name="llama2",
            cache_dir=temp_cache_dir / "testproject",
            project_name="testproject",
            auth_token="test-token",
            aws_region="us-east-1",
            tags=None,
            s3_bucket="my-custom-bucket",
            stage="dev",
            memory_size=8192,
            system_prompt=None,
        )

    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    def test_prepare_s3_bucket_immutable_error(
        self,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test that changing S3 bucket for existing deployment fails."""
        mock_get_config.return_value = temp_cache_dir

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)

        # Create existing zappa_settings.json with a bucket
        existing_settings = {
            "dev": {
                "s3_bucket": "existing-bucket",
                "app_function": "merle.app.app",
            }
        }
        (model_cache_dir / "zappa_settings.json").write_text(json.dumps(existing_settings))

        mock_get_cache.return_value = model_cache_dir

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket="different-bucket",  # Trying to change bucket
            stage="dev",
            memory_size=8192,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 1  # Should fail

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.get_default_project_name")
    def test_prepare_s3_bucket_same_as_existing(
        self,
        mock_get_default_project: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test that providing same S3 bucket as existing deployment succeeds."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_default_project.return_value = "testproject"

        model_cache_dir = temp_cache_dir / "testproject" / "llama2"
        model_cache_dir.mkdir(parents=True)

        # Create existing zappa_settings.json with a bucket
        existing_settings = {
            "dev": {
                "s3_bucket": "existing-bucket",
                "app_function": "merle.app.app",
            }
        }
        (model_cache_dir / "zappa_settings.json").write_text(json.dumps(existing_settings))

        mock_get_cache.return_value = model_cache_dir
        mock_prepare.return_value = model_cache_dir

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket="existing-bucket",  # Same bucket
            stage="dev",
            memory_size=8192,
            project=None,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 0
        mock_prepare.assert_called_once_with(
            model_name="llama2",
            cache_dir=temp_cache_dir / "testproject",
            project_name="testproject",
            auth_token="test-token",
            aws_region="us-east-1",
            tags=None,
            s3_bucket="existing-bucket",
            stage="dev",
            memory_size=8192,
            system_prompt=None,
        )

    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.get_default_project_name")
    def test_prepare_s3_bucket_reuses_existing(
        self,
        mock_get_default_project: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_prepare: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test that existing S3 bucket is reused when not specified."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_default_project.return_value = "testproject"

        model_cache_dir = temp_cache_dir / "testproject" / "llama2"
        model_cache_dir.mkdir(parents=True)

        # Create existing zappa_settings.json with a bucket
        existing_settings = {
            "dev": {
                "s3_bucket": "existing-bucket",
                "app_function": "merle.app.app",
            }
        }
        (model_cache_dir / "zappa_settings.json").write_text(json.dumps(existing_settings))

        mock_get_cache.return_value = model_cache_dir
        mock_prepare.return_value = model_cache_dir

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket=None,  # Not specified
            stage="dev",
            memory_size=8192,
            project=None,
        )

        result = handle_prepare_dockerfile(args)

        assert result == 0
        mock_prepare.assert_called_once_with(
            model_name="llama2",
            cache_dir=temp_cache_dir / "testproject",
            project_name="testproject",
            auth_token="test-token",
            aws_region="us-east-1",
            tags=None,
            s3_bucket="existing-bucket",  # Should reuse existing
            stage="dev",
            memory_size=8192,
            system_prompt=None,
        )


class TestHandleDeploy:
    """Tests for handle_deploy command."""

    @patch("boto3.client")
    @patch("merle.cli.subprocess.run")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_deploy_success_existing_files(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_subprocess: MagicMock,
        mock_boto_client: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test successful deploy with existing files."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)

        # Create valid zappa_settings.json with required fields
        zappa_settings = {
            "dev": {
                "aws_region": "us-east-1",
                "s3_bucket": "test-bucket",
            }
        }
        (model_cache_dir / "zappa_settings.json").write_text(json.dumps(zappa_settings))

        # Create config.json with auth token
        config = {"models": {"llama2": {"auth_token": "test-token"}}}
        (temp_cache_dir / "config.json").write_text(json.dumps(config))

        mock_get_cache.return_value = model_cache_dir

        # Mock ECR client
        mock_ecr = MagicMock()
        mock_ecr.create_repository.return_value = {}
        mock_ecr.describe_repositories.return_value = {
            "repositories": [{"repositoryUri": "123456789.dkr.ecr.us-east-1.amazonaws.com/merle-llama2"}]
        }
        # Mock ECR authentication with base64-encoded token
        mock_token = base64.b64encode(b"AWS:mock-password").decode()
        mock_ecr.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": mock_token,
                    "proxyEndpoint": "https://123456789.dkr.ecr.us-east-1.amazonaws.com",
                }
            ]
        }
        mock_boto_client.return_value = mock_ecr

        # Mock subprocess calls (docker login, docker build, docker push, zappa deploy)
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="mock-password")

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            stage="dev",
        )

        result = handle_deploy(args)

        assert result == 0
        # Should call subprocess multiple times: docker login, build, push, zappa deploy
        assert mock_subprocess.call_count >= 4

    @patch("boto3.client")
    @patch("merle.cli.subprocess.run")
    @patch("merle.cli.prepare_deployment_files")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_deploy_auto_prepare(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_prepare: MagicMock,
        mock_subprocess: MagicMock,
        mock_boto_client: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test deploy with auto-prepare when files don't exist."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        mock_get_cache.return_value = model_cache_dir
        mock_prepare.return_value = model_cache_dir

        # Create zappa_settings.json after prepare is called
        def create_settings(*args, **kwargs):
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            zappa_settings = {
                "dev": {
                    "aws_region": "us-east-1",
                    "s3_bucket": "test-bucket",
                }
            }
            (model_cache_dir / "zappa_settings.json").write_text(json.dumps(zappa_settings))

            # Create config.json with auth token
            config = {"models": {"llama2": {"auth_token": "test-token"}}}
            (temp_cache_dir / "config.json").write_text(json.dumps(config))
            return model_cache_dir

        mock_prepare.side_effect = create_settings

        # Mock ECR client
        mock_ecr = MagicMock()
        mock_ecr.create_repository.return_value = {}
        mock_ecr.describe_repositories.return_value = {
            "repositories": [{"repositoryUri": "123456789.dkr.ecr.us-east-1.amazonaws.com/merle-llama2"}]
        }
        # Mock ECR authentication with base64-encoded token
        mock_token = base64.b64encode(b"AWS:mock-password").decode()
        mock_ecr.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": mock_token,
                    "proxyEndpoint": "https://123456789.dkr.ecr.us-east-1.amazonaws.com",
                }
            ]
        }
        mock_boto_client.return_value = mock_ecr

        mock_subprocess.return_value = MagicMock(returncode=0, stdout="mock-password")

        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket=None,
            stage="dev",
            memory_size=8192,
        )

        result = handle_deploy(args)

        assert result == 0
        mock_prepare.assert_called_once()
        # Should call subprocess multiple times: docker login, build, push, zappa deploy
        assert mock_subprocess.call_count >= 4

    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_deploy_model_not_found(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test deploy when model cannot be determined."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.side_effect = ValueError("No models configured")

        args = argparse.Namespace(
            model=None,
            auth_token="test-token",
            region="us-east-1",
            cache_dir=None,
            tags=None,
            s3_bucket=None,
            stage="dev",
            memory_size=8192,
        )

        result = handle_deploy(args)

        assert result == 1


class TestHandleList:
    """Tests for handle_list command."""

    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.load_config")
    def test_list_no_models(self, mock_load_config: MagicMock, mock_get_config: MagicMock, temp_cache_dir: Path):
        """Test list command with no models configured."""
        mock_get_config.return_value = temp_cache_dir
        mock_load_config.return_value = {"models": {}}

        args = argparse.Namespace(
            cache_dir=None,
            raw=False,
            check_urls=False,
        )

        result = handle_list(args)

        assert result == 0

    @patch("merle.cli.get_deployment_url")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.load_config")
    def test_list_with_models(
        self,
        mock_load_config: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_get_url: MagicMock,
        temp_cache_dir: Path,
        sample_config: dict,
    ):
        """Test list command with configured models."""
        mock_get_config.return_value = temp_cache_dir
        mock_load_config.return_value = sample_config

        # Create model directories with zappa_settings.json
        for model_name in sample_config["models"]:
            model_dir = temp_cache_dir / model_name
            model_dir.mkdir(parents=True)
            (model_dir / "zappa_settings.json").write_text("{}")
            mock_get_cache.return_value = model_dir

        mock_get_url.return_value = None

        args = argparse.Namespace(
            cache_dir=None,
            raw=False,
            check_urls=False,
        )

        result = handle_list(args)

        assert result == 0

    @patch("merle.cli.mask_token")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_config_directory")
    @patch("merle.cli.load_config")
    def test_list_masks_tokens_by_default(
        self,
        mock_load_config: MagicMock,
        mock_get_config: MagicMock,
        mock_get_cache: MagicMock,
        mock_mask: MagicMock,
        temp_cache_dir: Path,
        sample_config: dict,
    ):
        """Test that tokens are masked by default."""
        mock_get_config.return_value = temp_cache_dir
        mock_load_config.return_value = sample_config
        mock_mask.return_value = "test...123"

        # Create model directory
        model_dir = temp_cache_dir / "llama2"
        model_dir.mkdir(parents=True)
        (model_dir / "zappa_settings.json").write_text("{}")
        mock_get_cache.return_value = model_dir

        args = argparse.Namespace(
            cache_dir=None,
            raw=False,
            check_urls=False,
        )

        result = handle_list(args)

        assert result == 0
        # Should call mask_token for each model
        assert mock_mask.call_count >= 1


class TestHandleDestroy:
    """Tests for handle_destroy command."""

    @patch("merle.cli.input")
    @patch("merle.cli.subprocess.run")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_destroy_success_with_confirmation(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_subprocess: MagicMock,
        mock_input: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test successful destroy with user confirmation."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "zappa_settings.json").write_text("{}")

        mock_get_cache.return_value = model_cache_dir
        mock_input.return_value = "yes"
        mock_subprocess.return_value = MagicMock(returncode=0)

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            yes=False,
            stage="dev",
        )

        result = handle_destroy(args)

        assert result == 0
        mock_input.assert_called_once()
        mock_subprocess.assert_called_once()

    @patch("merle.cli.subprocess.run")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_destroy_skip_confirmation(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_subprocess: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test destroy with --yes flag skips confirmation."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "zappa_settings.json").write_text("{}")

        mock_get_cache.return_value = model_cache_dir
        mock_subprocess.return_value = MagicMock(returncode=0)

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            yes=True,
            stage="dev",
        )

        result = handle_destroy(args)

        assert result == 0
        mock_subprocess.assert_called_once()

    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_destroy_no_deployment_found(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test destroy when no deployment exists."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        mock_get_cache.return_value = model_cache_dir

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            yes=True,
            stage="dev",
        )

        result = handle_destroy(args)

        assert result == 1

    @patch("merle.cli.save_config")
    @patch("merle.cli.load_config")
    @patch("merle.cli.subprocess.run")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_destroy_cleans_up_when_zappa_fails(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_subprocess: MagicMock,
        mock_load_config: MagicMock,
        mock_save_config: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test destroy cleans up local files even when zappa undeploy fails."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        # Create model cache directory and config
        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "zappa_settings.json").write_text("{}")

        # Create config file
        config_file = temp_cache_dir / "config.json"
        config_file.write_text('{"models": {"llama2": {"dev": {"region": "us-east-1"}}}}')

        mock_get_cache.return_value = model_cache_dir
        mock_load_config.return_value = {"models": {"llama2": {"dev": {"region": "us-east-1"}}}}

        # Simulate zappa undeploy failure (e.g., no AWS resources)
        mock_subprocess.return_value = MagicMock(returncode=1)

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            yes=True,
            stage="dev",
        )

        result = handle_destroy(args)

        # Should succeed (return 0) despite zappa failure
        assert result == 0

        # Verify cleanup still happened
        mock_save_config.assert_called_once()
        # Verify model was removed from config
        saved_config = mock_save_config.call_args[0][1]
        assert "llama2" not in saved_config.get("models", {})

        # Verify directory was cleaned up
        assert not model_cache_dir.exists()


class TestHandleChat:
    """Tests for handle_chat command."""

    @patch("merle.cli.run_interactive_chat")
    @patch("merle.cli.load_config")
    @patch("merle.cli.get_deployment_url")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_chat_success(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_get_url: MagicMock,
        mock_load_config: MagicMock,
        mock_run_chat: MagicMock,
        temp_cache_dir: Path,
        sample_config: dict,
    ):
        """Test successful chat command."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "zappa_settings.json").write_text("{}")

        mock_get_cache.return_value = model_cache_dir
        mock_get_url.return_value = "https://example.execute-api.us-east-1.amazonaws.com"
        mock_load_config.return_value = sample_config

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            stage="dev",
            debug=False,
        )

        result = handle_chat(args)

        assert result == 0
        mock_run_chat.assert_called_once_with(
            base_url="https://example.execute-api.us-east-1.amazonaws.com",
            auth_token="test-token-123",
            model="llama2",
            debug=False,
            system_prompt=None,
            context_window_size=None,
        )

    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_chat_model_not_prepared(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test chat when model is not prepared."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        mock_get_cache.return_value = model_cache_dir

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            stage="dev",
        )

        result = handle_chat(args)

        assert result == 1

    @patch("merle.cli.get_deployment_url")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_chat_model_not_deployed(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_get_url: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test chat when model is not deployed."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "zappa_settings.json").write_text("{}")

        mock_get_cache.return_value = model_cache_dir
        mock_get_url.return_value = None

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            stage="dev",
        )

        result = handle_chat(args)

        assert result == 1

    @patch("merle.cli.load_config")
    @patch("merle.cli.get_deployment_url")
    @patch("merle.cli.get_model_cache_dir")
    @patch("merle.cli.get_model_to_use")
    @patch("merle.cli.get_config_directory")
    def test_chat_no_auth_token(
        self,
        mock_get_config: MagicMock,
        mock_get_model: MagicMock,
        mock_get_cache: MagicMock,
        mock_get_url: MagicMock,
        mock_load_config: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test chat when no auth token is configured."""
        mock_get_config.return_value = temp_cache_dir
        mock_get_model.return_value = "llama2"

        model_cache_dir = temp_cache_dir / "llama2"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "zappa_settings.json").write_text("{}")

        mock_get_cache.return_value = model_cache_dir
        mock_get_url.return_value = "https://example.execute-api.us-east-1.amazonaws.com"
        mock_load_config.return_value = {"models": {"llama2": {"dev": {}}}}  # No auth_token

        args = argparse.Namespace(
            model="llama2",
            cache_dir=None,
            stage="dev",
        )

        result = handle_chat(args)

        assert result == 1
