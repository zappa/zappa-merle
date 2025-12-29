"""Tests for merle.functions module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from merle.functions import (
    generate_from_template,
    get_config_path,
    get_model_cache_dir,
    load_config,
    mask_token,
    normalize_model_name,
    parse_tags,
    prepare_deployment_files,
    save_config,
    update_model_config,
    validate_ollama_model,
)


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_normalize_forward_slash(self):
        """Test that forward slashes and hyphens are replaced with underscores."""
        result = normalize_model_name("schroneko/gemma-2-2b-jpn-it")
        assert result == "schroneko_gemma_2_2b_jpn_it"

    def test_normalize_multiple_special_chars(self):
        """Test normalization of multiple special characters."""
        result = normalize_model_name("owner/model:tag<>|")
        assert result == "owner_model_tag"

    def test_normalize_duplicate_underscores(self):
        """Test that duplicate underscores are collapsed."""
        result = normalize_model_name("model//name::tag")
        assert result == "model_name_tag"

    def test_normalize_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are removed."""
        result = normalize_model_name("/model/name/")
        assert result == "model_name"

    def test_normalize_simple_name(self):
        """Test that simple names without special chars remain unchanged."""
        result = normalize_model_name("llama2")
        assert result == "llama2"

    def test_normalize_huggingface_model_with_periods(self):
        """Test normalization of HuggingFace model names with periods in hf.co prefix."""
        result = normalize_model_name("hf.co/mmnga/cyberagent_deepseek_r1_distill_qwen_14b_japa")
        assert result == "hf_co_mmnga_cyberagent_deepseek_r1_distill_qwen_14b_japa"

    def test_normalize_model_with_multiple_periods(self):
        """Test that multiple periods are normalized correctly."""
        result = normalize_model_name("model.name.with.dots")
        assert result == "model_name_with_dots"


class TestMaskToken:
    """Tests for mask_token function."""

    def test_mask_normal_token(self):
        """Test masking a normal length token."""
        result = mask_token("abcdefghijklmnop", show_chars=4)
        assert result == "abcd...mnop"

    def test_mask_short_token(self):
        """Test masking a token shorter than show_chars * 2."""
        result = mask_token("abc", show_chars=4)
        assert result == "****"

    def test_mask_empty_token(self):
        """Test masking an empty token."""
        result = mask_token("", show_chars=4)
        assert result == "****"

    def test_mask_custom_show_chars(self):
        """Test masking with custom show_chars parameter."""
        result = mask_token("0123456789abcdef", show_chars=2)
        assert result == "01...ef"


class TestParseTags:
    """Tests for parse_tags function."""

    def test_parse_valid_tags(self):
        """Test parsing valid tag string."""
        result = parse_tags("Environment=dev,Project=ollama")
        assert result == {"Environment": "dev", "Project": "ollama"}

    def test_parse_tags_with_spaces(self):
        """Test parsing tags with spaces around equals and commas."""
        result = parse_tags("Environment = dev , Project = ollama")
        assert result == {"Environment": "dev", "Project": "ollama"}

    def test_parse_tags_with_equals_in_value(self):
        """Test parsing tags where value contains equals sign."""
        result = parse_tags("Key=value=with=equals")
        assert result == {"Key": "value=with=equals"}

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty dict."""
        result = parse_tags("")
        assert result == {}

    def test_parse_whitespace_string(self):
        """Test parsing whitespace-only string returns empty dict."""
        result = parse_tags("   ")
        assert result == {}

    def test_parse_invalid_format_no_equals(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tag format"):
            parse_tags("InvalidTag")

    def test_parse_invalid_empty_key(self):
        """Test that empty key raises ValueError."""
        with pytest.raises(ValueError, match="Tag key cannot be empty"):
            parse_tags("=value")

    def test_parse_multiple_tags_with_empty_pairs(self):
        """Test parsing with empty pairs between commas."""
        result = parse_tags("Key1=val1,,Key2=val2")
        assert result == {"Key1": "val1", "Key2": "val2"}


class TestConfigFunctions:
    """Tests for config-related functions."""

    def test_get_config_path(self, temp_cache_dir: Path):
        """Test getting config path."""
        config_path = get_config_path(temp_cache_dir)
        assert config_path == temp_cache_dir / "config.json"

    def test_save_and_load_config(self, temp_cache_dir: Path, sample_config: dict):
        """Test saving and loading config."""
        save_config(temp_cache_dir, sample_config)

        loaded_config = load_config(temp_cache_dir)
        assert loaded_config == sample_config

    def test_load_config_nonexistent_file(self, temp_cache_dir: Path):
        """Test loading config when file doesn't exist."""
        config = load_config(temp_cache_dir)
        assert config == {"models": {}}

    def test_load_config_invalid_json(self, temp_cache_dir: Path):
        """Test loading config with invalid JSON."""
        config_path = get_config_path(temp_cache_dir)
        config_path.write_text("invalid json content")

        config = load_config(temp_cache_dir)
        assert config == {"models": {}}

    def test_get_model_cache_dir(self, temp_cache_dir: Path):
        """Test getting model-specific cache directory."""
        model_dir = get_model_cache_dir(temp_cache_dir, "schroneko/gemma-2-2b-jpn-it")
        assert model_dir == temp_cache_dir / "dev" / "schroneko_gemma_2_2b_jpn_it"

    def test_update_model_config_new_model(self, temp_cache_dir: Path):
        """Test updating config for a new model."""
        update_model_config(
            cache_dir=temp_cache_dir,
            model_name="llama2",
            auth_token="token123",
            region="us-east-1",
        )

        config = load_config(temp_cache_dir)
        assert "llama2" in config["models"]
        assert config["models"]["llama2"]["dev"]["auth_token"] == "token123"
        assert config["models"]["llama2"]["dev"]["region"] == "us-east-1"
        assert config["models"]["llama2"]["dev"]["normalized_name"] == "llama2"

    def test_update_model_config_with_tags(self, temp_cache_dir: Path, sample_tags: dict):
        """Test updating config with tags."""
        update_model_config(
            cache_dir=temp_cache_dir,
            model_name="llama2",
            tags=sample_tags,
        )

        config = load_config(temp_cache_dir)
        assert config["models"]["llama2"]["dev"]["tags"] == sample_tags

    def test_update_model_config_existing_model(self, temp_cache_dir: Path, sample_config: dict):
        """Test updating config for existing model."""
        save_config(temp_cache_dir, sample_config)

        update_model_config(
            cache_dir=temp_cache_dir,
            model_name="llama2",
            auth_token="new-token",
        )

        config = load_config(temp_cache_dir)
        assert config["models"]["llama2"]["dev"]["auth_token"] == "new-token"


class TestValidateOllamaModel:
    """Tests for validate_ollama_model function."""

    @patch("merle.functions.httpx.get")
    def test_validate_model_api_success(self, mock_get: MagicMock):
        """Test successful model validation via API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = validate_ollama_model("llama2")
        assert result is True

    @patch("merle.functions.httpx.get")
    def test_validate_model_with_owner(self, mock_get: MagicMock):
        """Test validating model name with owner/model format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = validate_ollama_model("schroneko/gemma-2-2b-jpn-it")
        assert result is True

    @patch("merle.functions.httpx.get")
    def test_validate_model_invalid_format_too_many_slashes(self, mock_get: MagicMock):
        """Test that invalid model format raises ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid model name format"):
            validate_ollama_model("owner/model/extra")

    def test_validate_model_empty_name(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            validate_ollama_model("")

    @patch("merle.functions.httpx.get")
    def test_validate_model_timeout(self, mock_get: MagicMock):
        """Test model validation with timeout."""
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        # Should still pass with basic format validation
        result = validate_ollama_model("llama2")
        assert result is True

    @patch("merle.functions.httpx.get")
    def test_validate_model_timeout_invalid_format(self, mock_get: MagicMock):
        """Test model validation timeout with invalid format."""
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(ValueError, match="Invalid model name format"):
            validate_ollama_model("owner/model/extra")


class TestGenerateFromTemplate:
    """Tests for generate_from_template function."""

    def test_generate_simple_template(self, tmp_path: Path):
        """Test generating file from simple template."""
        template_path = tmp_path / "template.txt"
        template_path.write_text("Hello {{NAME}}, welcome to {{PLACE}}!")

        output_path = tmp_path / "output.txt"
        replacements = {"NAME": "Alice", "PLACE": "Wonderland"}

        generate_from_template(template_path, output_path, replacements)

        assert output_path.exists()
        assert output_path.read_text() == "Hello Alice, welcome to Wonderland!"

    def test_generate_json_template(self, tmp_path: Path):
        """Test generating JSON file from template."""
        template_path = tmp_path / "template.json"
        template_path.write_text('{"model": "{{MODEL}}", "region": "{{REGION}}"}')

        output_path = tmp_path / "output.json"
        replacements = {"MODEL": "llama2", "REGION": "us-east-1"}

        generate_from_template(template_path, output_path, replacements)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data == {"model": "llama2", "region": "us-east-1"}

    def test_generate_template_nonexistent_file(self, tmp_path: Path):
        """Test that nonexistent template raises FileNotFoundError."""
        template_path = tmp_path / "nonexistent.txt"
        output_path = tmp_path / "output.txt"

        with pytest.raises(FileNotFoundError, match="Template file not found"):
            generate_from_template(template_path, output_path, {})

    def test_generate_template_creates_parent_dirs(self, tmp_path: Path):
        """Test that output parent directories are created."""
        template_path = tmp_path / "template.txt"
        template_path.write_text("Content: {{VALUE}}")

        output_path = tmp_path / "subdir" / "nested" / "output.txt"
        replacements = {"VALUE": "test"}

        generate_from_template(template_path, output_path, replacements)

        assert output_path.exists()
        assert output_path.read_text() == "Content: test"


class TestGenerateZappaSettings:
    """Tests for _generate_zappa_settings function."""

    def test_generate_zappa_settings_basic(self, tmp_path: Path):
        """Test generating zappa settings with basic configuration."""
        from merle.functions import _generate_zappa_settings  # noqa: PLC0415

        output_path = tmp_path / "zappa_settings.json"
        _generate_zappa_settings(
            output_path=output_path,
            model_name="llama2",
            aws_region="us-east-1",
            tags={},
            s3_bucket="test-zappa-bucket",
            auth_token="test-auth-token-12345",
            project_name="myproject",
        )

        assert output_path.exists()
        with output_path.open() as f:
            settings = json.load(f)

        # Verify structure
        assert "dev" in settings
        dev_config = settings["dev"]

        # Verify core configuration
        assert dev_config["app_function"] == "merle.app.app"
        assert dev_config["project_name"] == "myproject"  # Uses normalized project name only
        assert dev_config["s3_bucket"] == "test-zappa-bucket"
        assert dev_config["aws_region"] == "us-east-1"
        assert dev_config["memory_size"] == 8192
        assert dev_config["timeout_seconds"] == 900

        # Verify environment variables
        env_vars = dev_config["environment_variables"]
        assert env_vars["OLLAMA_MODEL"] == "llama2"
        assert env_vars["OLLAMA_URL"] == "http://localhost:11434"
        assert env_vars["OLLAMA_MODELS"] == "/tmp/models"  # noqa: S108
        assert env_vars["OLLAMA_STARTUP_TIMEOUT"] == "120"
        assert env_vars["ZAPPA_RUNNING_IN_DOCKER"] == "True"
        assert env_vars["API_KEY"] == "test-auth-token-12345"

        # Verify other settings
        assert dev_config["ephemeral_storage"] == {"Size": 5120}
        assert dev_config["keep_warm"] is False
        assert dev_config["keep_warm_expression"] == "rate(4 minutes)"
        assert dev_config["cors"] is True
        assert dev_config["cors_allow_headers"] == ["Content-Type", "X-API-Key"]
        assert dev_config["binary_support"] is False
        assert dev_config["tags"] == {}

        # Verify authorizer
        authorizer = dev_config["authorizer"]
        assert authorizer["function"] == "authorizer.lambda_handler"
        assert authorizer["token_header"] == "X-API-Key"
        assert authorizer["result_ttl"] == 300

    def test_generate_zappa_settings_with_tags(self, tmp_path: Path, sample_tags: dict):
        """Test generating zappa settings with tags."""
        from merle.functions import _generate_zappa_settings  # noqa: PLC0415

        output_path = tmp_path / "zappa_settings.json"
        _generate_zappa_settings(
            output_path=output_path,
            model_name="llama2",
            aws_region="us-west-2",
            tags=sample_tags,
            s3_bucket="test-bucket-with-tags",
            auth_token="test-auth-token-67890",
            project_name="myproject",
        )

        with output_path.open() as f:
            settings = json.load(f)

        assert settings["dev"]["tags"] == sample_tags
        assert settings["dev"]["aws_region"] == "us-west-2"
        assert settings["dev"]["s3_bucket"] == "test-bucket-with-tags"


class TestPrepareDeploymentFiles:
    """Tests for prepare_deployment_files function."""

    @patch("merle.functions.validate_ollama_model")
    def test_prepare_deployment_files_basic(self, mock_validate: MagicMock, temp_cache_dir: Path):
        """Test preparing deployment files with basic parameters."""
        mock_validate.return_value = True

        # Create template directory with mock templates
        template_dir = Path(__file__).parent.parent / "merle" / "templates"

        model_cache_dir = prepare_deployment_files(
            model_name="llama2",
            cache_dir=temp_cache_dir,
            project_name="testproject",
            auth_token="test-token",
            aws_region="us-east-1",
            s3_bucket="test-bucket",
        )

        # Verify model cache directory was created
        assert model_cache_dir.exists()
        assert model_cache_dir.name == "llama2"

        # Verify files were generated
        assert (model_cache_dir / "Dockerfile").exists()
        assert (model_cache_dir / "zappa_settings.json").exists()
        assert (model_cache_dir / "authorizer.py").exists()

        # Verify config was updated
        config = load_config(temp_cache_dir)
        assert "llama2" in config["models"]
        assert config["models"]["llama2"]["dev"]["auth_token"] == "test-token"
        assert config["models"]["llama2"]["dev"]["region"] == "us-east-1"

    @patch("merle.functions.validate_ollama_model")
    def test_prepare_deployment_files_with_tags(
        self, mock_validate: MagicMock, temp_cache_dir: Path, sample_tags: dict
    ):
        """Test preparing deployment files with tags."""
        mock_validate.return_value = True

        model_cache_dir = prepare_deployment_files(
            model_name="llama2",
            cache_dir=temp_cache_dir,
            project_name="testproject",
            auth_token="test-token",
            tags=sample_tags,
            s3_bucket="test-bucket-tags",
        )

        # Verify zappa_settings.json contains tags
        zappa_settings_path = model_cache_dir / "zappa_settings.json"
        with zappa_settings_path.open() as f:
            settings = json.load(f)

        assert "tags" in settings["dev"]
        assert settings["dev"]["tags"] == sample_tags

        # Verify config contains tags
        config = load_config(temp_cache_dir)
        assert config["models"]["llama2"]["dev"]["tags"] == sample_tags

    @patch("merle.functions.validate_ollama_model")
    def test_prepare_deployment_files_without_tags(self, mock_validate: MagicMock, temp_cache_dir: Path):
        """Test preparing deployment files without tags."""
        mock_validate.return_value = True

        model_cache_dir = prepare_deployment_files(
            model_name="llama2",
            cache_dir=temp_cache_dir,
            project_name="testproject",
            auth_token="test-token-no-tags",
            s3_bucket="test-bucket-no-tags",
        )

        # Verify zappa_settings.json has empty tags
        zappa_settings_path = model_cache_dir / "zappa_settings.json"
        with zappa_settings_path.open() as f:
            settings = json.load(f)

        assert settings["dev"]["tags"] == {}

        # Verify config doesn't have tags key
        config = load_config(temp_cache_dir)
        assert "tags" not in config["models"]["llama2"]["dev"]

    @patch("merle.functions.validate_ollama_model")
    def test_prepare_deployment_files_invalid_model(self, mock_validate: MagicMock, temp_cache_dir: Path):
        """Test that invalid model name raises ValueError."""
        mock_validate.side_effect = ValueError("Invalid model name")

        with pytest.raises(ValueError, match="Invalid model name"):
            prepare_deployment_files(
                model_name="invalid///model",
                cache_dir=temp_cache_dir,
                project_name="testproject",
                s3_bucket="test-bucket",
            )

    @patch("merle.functions.validate_ollama_model")
    def test_prepare_deployment_files_normalized_directory(self, mock_validate: MagicMock, temp_cache_dir: Path):
        """Test that model directory is normalized."""
        mock_validate.return_value = True

        model_cache_dir = prepare_deployment_files(
            model_name="owner/model-name",
            cache_dir=temp_cache_dir,
            project_name="testproject",
            auth_token="test-token-normalized",
            s3_bucket="test-bucket-normalized",
            skip_model_download=True,
        )

        assert model_cache_dir.name == "owner_model_name"
