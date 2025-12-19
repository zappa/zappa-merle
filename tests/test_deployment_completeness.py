"""Tests to verify deployment completeness and identify missing components."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from merle.cli import handle_prepare_dockerfile


class TestDeploymentCompleteness:
    """Test that verifies what's missing for successful deployment."""

    @patch("merle.functions.validate_ollama_model")
    def test_prepare_creates_expected_files(self, mock_validate: MagicMock, temp_cache_dir: Path) -> None:
        """Test that prepare command creates expected deployment files."""
        mock_validate.return_value = True

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
        )

        result = handle_prepare_dockerfile(args)
        assert result == 0

        # Verify expected files were created
        model_dir = temp_cache_dir / "dev" / "llama2"
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
        config_path = temp_cache_dir / "config.json"
        assert config_path.exists(), "config.json should be created"

    @patch("merle.functions.validate_ollama_model")
    def test_deployment_is_now_complete(  # noqa: PLR0915, PLR0912
        self, mock_validate: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Verify that deployment is now complete with all required components."""
        mock_validate.return_value = True

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
        )

        result = handle_prepare_dockerfile(args)
        assert result == 0, "Prepare command should succeed"

        model_dir = temp_cache_dir / "dev" / "llama2"
        project_root = Path(__file__).parent.parent

        # Verify all components are now present
        checks_passed = []
        checks_failed = []

        # Check 1: app.py exists in source
        if (project_root / "merle" / "app.py").exists():
            checks_passed.append("✓ merle/app.py exists in source")
        else:
            checks_failed.append("✗ merle/app.py missing in source")

        # Check 2: app.py copied to cache
        if (model_dir / "merle" / "app.py").exists():
            checks_passed.append("✓ merle/app.py copied to cache directory")
        else:
            checks_failed.append("✗ merle/app.py not copied to cache")

        # Check 3: Flask dependency
        pyproject_path = project_root / "pyproject.toml"
        with pyproject_path.open() as f:
            if "flask" in f.read().lower():
                checks_passed.append("✓ Flask dependency in pyproject.toml")
            else:
                checks_failed.append("✗ Flask dependency missing")

        # Check 4: Zappa dependency
        with pyproject_path.open() as f:
            if "zappa" in f.read().lower():
                checks_passed.append("✓ Zappa dependency in pyproject.toml")
            else:
                checks_failed.append("✗ Zappa dependency missing")

        # Check 5: pyproject.toml copied
        if (model_dir / "pyproject.toml").exists():
            checks_passed.append("✓ pyproject.toml copied to cache")
        else:
            checks_failed.append("✗ pyproject.toml not copied")

        # Check 6: merle/ directory copied
        if (model_dir / "merle").exists():
            checks_passed.append("✓ merle/ directory copied to cache")
        else:
            checks_failed.append("✗ merle/ directory not copied")

        # Check 7: Dockerfile generated
        if (model_dir / "Dockerfile").exists():
            checks_passed.append("✓ Dockerfile generated")
        else:
            checks_failed.append("✗ Dockerfile not generated")

        # Check 8: zappa_settings.json generated
        if (model_dir / "zappa_settings.json").exists():
            checks_passed.append("✓ zappa_settings.json generated")
        else:
            checks_failed.append("✗ zappa_settings.json not generated")

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
