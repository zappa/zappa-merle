"""Tests for merle.managers module."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from merle.managers import DeploymentManager, _subprocess_env_with_venv_bin


@pytest.fixture
def manager(tmp_path: Path) -> DeploymentManager:
    """Create a DeploymentManager instance for testing."""
    return DeploymentManager(
        model_name="llama2",
        cache_dir=tmp_path / "cache",
        project_name="test-project",
        stage="dev",
        region="us-east-1",
    )


@pytest.fixture
def prepared_manager(manager: DeploymentManager) -> DeploymentManager:
    """Create a DeploymentManager with prepared deployment files."""
    manager.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "dev": {
            "app_function": "merle.app.app",
            "project_name": "test-project",
            "s3_bucket": "test-bucket",
            "aws_region": "us-east-1",
            "memory_size": 8192,
            "timeout_seconds": 900,
            "docker_image_uri": "123456789.dkr.ecr.us-east-1.amazonaws.com/merle-llama2:latest",
        }
    }
    manager.zappa_settings_path.write_text(json.dumps(settings))
    return manager


class TestDeploymentManagerProperties:
    """Tests for DeploymentManager property methods."""

    def test_model_cache_dir(self, manager):
        cache_dir = manager.model_cache_dir
        assert isinstance(cache_dir, Path)
        assert "llama2" in str(cache_dir)

    def test_zappa_settings_path(self, manager):
        path = manager.zappa_settings_path
        assert path.name == "zappa_settings.json"

    def test_is_prepared_false(self, manager):
        assert manager.is_prepared is False

    def test_is_prepared_true(self, prepared_manager):
        assert prepared_manager.is_prepared is True

    def test_normalized_model_name(self, manager):
        assert manager.normalized_model_name == "llama2"

    def test_normalized_model_name_with_slashes(self, tmp_path):
        mgr = DeploymentManager(
            model_name="user/model:tag",
            cache_dir=tmp_path,
            project_name="proj",
        )
        assert "/" not in mgr.normalized_model_name
        assert ":" not in mgr.normalized_model_name

    def test_ecr_repo_name(self, manager):
        assert manager.ecr_repo_name == "merle-llama2"

    def test_region_defaults_to_settings(self, tmp_path):
        with patch("merle.managers.REGION", "ap-northeast-1"):
            mgr = DeploymentManager(
                model_name="llama2",
                cache_dir=tmp_path,
                project_name="proj",
            )
            assert mgr.region == "ap-northeast-1"

    def test_region_override(self, manager):
        assert manager.region == "us-east-1"


class TestCalculateEphemeralStorage:
    """Tests for _calculate_ephemeral_storage."""

    def test_small_model_gets_minimum(self, manager):
        result = manager._calculate_ephemeral_storage(0.0)
        assert result == 512

    def test_large_model_gets_clamped(self, manager):
        result = manager._calculate_ephemeral_storage(20.0)
        assert result == 10240

    def test_medium_model_rounds_up(self, manager):
        result = manager._calculate_ephemeral_storage(3.0)
        assert result % 512 == 0
        assert result >= 3 * 1024  # at least the model size in MB


class TestGetAuthorizerLambdaName:
    """Tests for _get_authorizer_lambda_name."""

    def test_name_format(self, manager):
        name = manager._get_authorizer_lambda_name()
        assert name.endswith("-dev-authorizer")
        assert "test" in name.lower()

    def test_name_with_different_stage(self, tmp_path):
        mgr = DeploymentManager(
            model_name="llama2",
            cache_dir=tmp_path,
            project_name="myproj",
            stage="prod",
        )
        name = mgr._get_authorizer_lambda_name()
        assert name.endswith("-prod-authorizer")


class TestGenerateZappaSettings:
    """Tests for _generate_zappa_settings."""

    @patch("merle.managers.ZappaCLI")
    def test_generates_settings_file(self, mock_zappa_cli_cls, manager):
        mock_cli = MagicMock()
        mock_cli._generate_settings_dict.return_value = {"dev": {}}
        mock_zappa_cli_cls.return_value = mock_cli

        manager.model_cache_dir.mkdir(parents=True, exist_ok=True)
        manager._generate_zappa_settings(
            auth_token="test-token",
            s3_bucket="my-bucket",
            tags={"Env": "dev"},
            memory_size=8192,
        )

        assert manager.zappa_settings_path.exists()
        settings = json.loads(manager.zappa_settings_path.read_text())
        assert "dev" in settings
        assert settings["dev"]["app_function"] == "merle.app.app"
        assert settings["dev"]["s3_bucket"] == "my-bucket"
        assert settings["dev"]["memory_size"] == 8192

    @patch("merle.managers.ZappaCLI")
    def test_split_mode_adds_s3_permissions(self, mock_zappa_cli_cls, manager):
        mock_cli = MagicMock()
        mock_cli._generate_settings_dict.return_value = {"dev": {}}
        mock_zappa_cli_cls.return_value = mock_cli

        manager.model_cache_dir.mkdir(parents=True, exist_ok=True)
        manager._generate_zappa_settings(
            auth_token="test-token",
            s3_bucket="my-bucket",
            tags={},
            use_split=True,
        )

        settings = json.loads(manager.zappa_settings_path.read_text())
        assert "extra_permissions" in settings["dev"]
        assert settings["dev"]["environment_variables"]["MERLE_SPLIT_MODEL"] == "true"

    @patch("merle.managers.ZappaCLI")
    def test_context_window_env_var(self, mock_zappa_cli_cls, manager):
        mock_cli = MagicMock()
        mock_cli._generate_settings_dict.return_value = {"dev": {}}
        mock_zappa_cli_cls.return_value = mock_cli

        manager.model_cache_dir.mkdir(parents=True, exist_ok=True)
        manager._generate_zappa_settings(
            auth_token="tok",
            s3_bucket="bkt",
            tags={},
            context_window_size=4096,
        )

        settings = json.loads(manager.zappa_settings_path.read_text())
        assert settings["dev"]["environment_variables"]["OLLAMA_MODEL_CONTEXT_WINDOW_SIZE"] == "4096"

    @patch("merle.managers.ZappaCLI")
    def test_authorizer_arn_used_when_provided(self, mock_zappa_cli_cls, manager):
        mock_cli = MagicMock()
        mock_cli._generate_settings_dict.return_value = {"dev": {}}
        mock_zappa_cli_cls.return_value = mock_cli

        manager.model_cache_dir.mkdir(parents=True, exist_ok=True)
        manager._generate_zappa_settings(
            auth_token="tok",
            s3_bucket="bkt",
            tags={},
            authorizer_arn="arn:aws:lambda:us-east-1:123:function:auth",
        )

        settings = json.loads(manager.zappa_settings_path.read_text())
        assert settings["dev"]["authorizer"]["arn"] == "arn:aws:lambda:us-east-1:123:function:auth"


class TestUpdateZappaSettingsWithImageUri:
    """Tests for _update_zappa_settings_with_image_uri."""

    def test_adds_docker_image_uri(self, prepared_manager):
        new_uri = "999.dkr.ecr.us-east-1.amazonaws.com/merle-llama2:v2"
        prepared_manager._update_zappa_settings_with_image_uri(new_uri)

        settings = json.loads(prepared_manager.zappa_settings_path.read_text())
        assert settings["dev"]["docker_image_uri"] == new_uri


class TestUpdateZappaSettingsWithAuthorizer:
    """Tests for _update_zappa_settings_with_authorizer."""

    def test_updates_authorizer_config(self, prepared_manager):
        arn = "arn:aws:lambda:us-east-1:123:function:my-auth"
        prepared_manager._update_zappa_settings_with_authorizer(arn)

        settings = json.loads(prepared_manager.zappa_settings_path.read_text())
        assert settings["dev"]["authorizer"]["arn"] == arn
        assert settings["dev"]["authorizer"]["token_header"] == "X-API-Key"
        assert settings["dev"]["authorizer"]["result_ttl"] == 300


class TestBuildAndPushDockerImage:
    """Tests for build_and_push_docker_image."""

    def test_raises_if_not_prepared(self, manager):
        with pytest.raises(RuntimeError, match="not prepared"):
            manager.build_and_push_docker_image()

    @patch("merle.managers.subprocess.run")
    @patch("merle.managers.boto3.client")
    def test_creates_ecr_repo_and_pushes(self, mock_boto_client, mock_run, prepared_manager):
        mock_ecr = MagicMock()
        mock_ecr.describe_repositories.return_value = {
            "repositories": [{"repositoryUri": "123.dkr.ecr.us-east-1.amazonaws.com/merle-llama2"}]
        }
        mock_ecr.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": "dXNlcjpwYXNz",  # base64("user:pass")
                    "proxyEndpoint": "https://123.dkr.ecr.us-east-1.amazonaws.com",
                }
            ]
        }
        mock_ecr.exceptions = MagicMock()
        mock_ecr.exceptions.RepositoryAlreadyExistsException = type("RepoExists", (Exception,), {})
        mock_boto_client.return_value = mock_ecr

        mock_run.return_value = MagicMock(returncode=0)

        image_uri = prepared_manager.build_and_push_docker_image()
        assert "merle-llama2:latest" in image_uri
        assert mock_run.call_count == 3  # login, build, push


class TestDeploy:
    """Tests for deploy."""

    def test_raises_if_not_prepared(self, manager):
        with pytest.raises(RuntimeError, match="not prepared"):
            manager.deploy(auth_token="test")

    @patch("merle.managers.subprocess.run")
    @patch("merle.managers.boto3.client")
    @patch.object(DeploymentManager, "_get_or_create_authorizer_role", return_value="arn:aws:iam::123:role/auth-role")
    @patch.object(
        DeploymentManager,
        "_deploy_authorizer_lambda",
        return_value="arn:aws:lambda:us-east-1:123:function:auth",
    )
    @patch.object(
        DeploymentManager, "get_deployment_url", return_value="https://example.execute-api.us-east-1.amazonaws.com/dev"
    )
    @patch("merle.managers.update_model_config")
    def test_deploys_with_zappa(
        self,
        mock_update_config,
        mock_get_url,
        mock_deploy_auth,
        mock_create_role,
        mock_boto_client,
        mock_run,
        prepared_manager,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="Deployed!", stderr="")

        url = prepared_manager.deploy(auth_token="test-token")
        assert url is not None
        mock_run.assert_called_once()
        mock_deploy_auth.assert_called_once()


class TestDestroy:
    """Tests for destroy."""

    def test_returns_false_if_not_prepared(self, manager):
        assert manager.destroy() is False

    @patch("merle.managers.subprocess.run")
    @patch.object(DeploymentManager, "_delete_authorizer_lambda")
    @patch.object(DeploymentManager, "_delete_authorizer_role")
    @patch.object(DeploymentManager, "_cleanup_local_files")
    def test_calls_zappa_undeploy(self, mock_cleanup, mock_del_role, mock_del_lambda, mock_run, prepared_manager):
        mock_run.return_value = MagicMock(returncode=0)

        result = prepared_manager.destroy()
        assert result is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "undeploy" in cmd
        assert "--yes" in cmd
        mock_del_lambda.assert_called_once()
        mock_del_role.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch("merle.managers.subprocess.run")
    @patch.object(DeploymentManager, "_delete_authorizer_lambda")
    @patch.object(DeploymentManager, "_delete_authorizer_role")
    @patch.object(DeploymentManager, "_cleanup_local_files")
    def test_returns_false_on_zappa_failure(
        self, mock_cleanup, mock_del_role, mock_del_lambda, mock_run, prepared_manager
    ):
        mock_run.return_value = MagicMock(returncode=1)

        result = prepared_manager.destroy()
        assert result is False
        # Cleanup should still happen
        mock_del_lambda.assert_called_once()
        mock_del_role.assert_called_once()
        mock_cleanup.assert_called_once()


class TestSubprocessEnvWithVenvBin:
    """Tests for _subprocess_env_with_venv_bin (zappa PATH fix for Issue #4)."""

    def test_prepends_sys_executable_bin_to_path(self):
        env = _subprocess_env_with_venv_bin()
        expected_bin = str(Path(sys.executable).parent)
        assert env["PATH"].startswith(f"{expected_bin}{os.pathsep}")

    def test_merges_extra_env_variables(self):
        env = _subprocess_env_with_venv_bin({"API_KEY": "tok", "FOO": "bar"})
        assert env["API_KEY"] == "tok"
        assert env["FOO"] == "bar"

    def test_preserves_existing_path_entries(self, monkeypatch):
        monkeypatch.setenv("PATH", "/existing/bin")
        env = _subprocess_env_with_venv_bin()
        assert "/existing/bin" in env["PATH"]


class TestPrepareGeneratesMinimalPyproject:
    """Preparation should emit a minimal pyproject, not copy the consumer's (Issue #3)."""

    @patch("merle.managers.ZappaCLI")
    @patch("merle.managers.update_model_config")
    @patch("merle.managers.validate_ollama_model", return_value=True)
    def test_generates_template_pyproject_with_merle_dependency(
        self,
        mock_validate,
        mock_update,
        mock_zappa_cli_cls,
        manager,
    ):
        mock_cli = MagicMock()
        mock_cli._generate_settings_dict.return_value = {"dev": {}}
        mock_zappa_cli_cls.return_value = mock_cli

        manager.prepare(auth_token="tok", s3_bucket="bkt", skip_model_download=True)

        pyproject = (manager.model_cache_dir / "pyproject.toml").read_text()
        assert "merle @ git+https://github.com/zappa/zappa-merle.git" in pyproject
        assert "-runtime" in pyproject
        mock_validate.assert_called_once()
        mock_update.assert_called_once()


class TestCleanupLocalFiles:
    """Tests for _cleanup_local_files."""

    @patch("merle.functions.save_config")
    @patch("merle.functions.load_config")
    def test_removes_model_from_config(self, mock_load, mock_save, prepared_manager):
        mock_load.return_value = {
            "models": {
                "llama2": {
                    "dev": {"auth_token": "tok"},
                }
            }
        }

        prepared_manager._cleanup_local_files()

        saved_config = mock_save.call_args[0][1]
        assert "llama2" not in saved_config.get("models", {})

    @patch("merle.functions.save_config")
    @patch("merle.functions.load_config")
    def test_deletes_cache_directory(self, mock_load, mock_save, prepared_manager):
        mock_load.return_value = {"models": {}}
        cache_dir = prepared_manager.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "somefile.txt").write_text("test")

        prepared_manager._cleanup_local_files()
        assert not cache_dir.exists()
