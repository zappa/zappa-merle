"""Tests to verify deployment completeness and identify missing components."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from merle.cli import handle_prepare_dockerfile


class TestDeploymentCompleteness:
    """Test that verifies what's missing for successful deployment."""

    @patch("merle.model_split.get_ollama_models_dir")
    @patch("merle.functions.validate_ollama_model")
    @patch("merle.cli.get_default_project_name")
    def test_prepare_creates_expected_files(
        self,
        mock_get_default_project: MagicMock,
        mock_validate: MagicMock,
        mock_models_dir_fn: MagicMock,
        temp_cache_dir: Path,
        mock_models_dir: Path,
    ) -> None:
        """Test that prepare command creates expected deployment files."""
        mock_validate.return_value = True
        mock_get_default_project.return_value = "testproject"
        mock_models_dir_fn.return_value = mock_models_dir

        # Prepare deployment files
        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=str(temp_cache_dir),
            tags=None,
            stage="dev",
            memory_size=8192,
            s3_bucket=None,
            project=None,
        )

        result = handle_prepare_dockerfile(args)
        assert result == 0

        # Verify expected files were created
        model_dir = temp_cache_dir / "testproject" / "dev" / "llama2"
        assert model_dir.exists(), "Model directory should be created"

        expected_files = [
            "Dockerfile",
            "zappa_settings.json",
            "authorizer.py",
        ]

        for filename in expected_files:
            file_path = model_dir / filename
            assert file_path.exists(), f"{filename} should be created"

        # Verify config.json was created
        config_path = temp_cache_dir / "testproject" / "config.json"
        assert config_path.exists(), "config.json should be created"

    @patch("merle.model_split.get_ollama_models_dir")
    @patch("merle.functions.validate_ollama_model")
    @patch("merle.cli.get_default_project_name")
    def test_deployment_is_now_complete(  # noqa: PLR0915, PLR0912
        self,
        mock_get_default_project: MagicMock,
        mock_validate: MagicMock,
        mock_models_dir_fn: MagicMock,
        temp_cache_dir: Path,
        mock_models_dir: Path,
    ) -> None:
        """Verify that deployment is now complete with all required components."""
        mock_validate.return_value = True
        mock_get_default_project.return_value = "testproject"
        mock_models_dir_fn.return_value = mock_models_dir

        # Prepare deployment files
        args = argparse.Namespace(
            model="llama2",
            auth_token="test-token",
            region="us-east-1",
            cache_dir=str(temp_cache_dir),
            tags=None,
            stage="dev",
            memory_size=8192,
            s3_bucket=None,
            project=None,
        )

        result = handle_prepare_dockerfile(args)
        assert result == 0, "Prepare command should succeed"

        model_dir = temp_cache_dir / "testproject" / "dev" / "llama2"
        project_root = Path(__file__).parent.parent

        # Verify all components are now present
        # Note: merle/ package is installed during Docker build via uv sync, not copied
        checks_passed = []
        checks_failed = []

        # Check 1: app.py exists in source
        if (project_root / "merle" / "app.py").exists():
            checks_passed.append("✓ merle/app.py exists in source")
        else:
            checks_failed.append("✗ merle/app.py missing in source")

        # Check 2: Flask dependency
        pyproject_path = project_root / "pyproject.toml"
        with pyproject_path.open() as f:
            if "flask" in f.read().lower():
                checks_passed.append("✓ Flask dependency in pyproject.toml")
            else:
                checks_failed.append("✗ Flask dependency missing")

        # Check 3: Zappa dependency
        with pyproject_path.open() as f:
            if "zappa" in f.read().lower():
                checks_passed.append("✓ Zappa dependency in pyproject.toml")
            else:
                checks_failed.append("✗ Zappa dependency missing")

        # Check 4: pyproject.toml copied
        if (model_dir / "pyproject.toml").exists():
            checks_passed.append("✓ pyproject.toml copied to cache")
        else:
            checks_failed.append("✗ pyproject.toml not copied")

        # Check 5: Dockerfile generated
        if (model_dir / "Dockerfile").exists():
            checks_passed.append("✓ Dockerfile generated")
        else:
            checks_failed.append("✗ Dockerfile not generated")

        # Check 6: zappa_settings.json generated
        if (model_dir / "zappa_settings.json").exists():
            checks_passed.append("✓ zappa_settings.json generated")
        else:
            checks_failed.append("✗ zappa_settings.json not generated")

        # Check 7: models/ directory with pre-downloaded model
        if (model_dir / "models").exists():
            checks_passed.append("✓ models/ directory created")
        else:
            checks_failed.append("✗ models/ directory not created")

        # Print summary
        print("\n" + "=" * 80)  # noqa: T201
        print("DEPLOYMENT READINESS VERIFICATION:")  # noqa: T201
        print("=" * 80)  # noqa: T201
        for check in checks_passed:
            print(check)  # noqa: T201
        if checks_failed:
            for check in checks_failed:
                print(check)  # noqa: T201
        print("=" * 80)  # noqa: T201
        print(f"Status: {len(checks_passed)}/{len(checks_passed) + len(checks_failed)} checks passed")  # noqa: T201
        print("=" * 80 + "\n")  # noqa: T201

        # Assert all checks passed
        assert len(checks_failed) == 0, (
            "Deployment readiness checks failed:\n"
            + "\n".join(checks_failed)
            + "\n\nAll components should now be present for successful deployment."
        )
