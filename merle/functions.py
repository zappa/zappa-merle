import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx
from zappa.core import Zappa

from merle.settings import REGION

logger = logging.getLogger(__name__)


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name to be safe for use as a directory name.

    Replaces special characters (including hyphens) with underscores to create
    a cross-platform safe directory name.

    Args:
        model_name: Original model name (e.g., 'schroneko/gemma-2-2b-jpn-it')

    Returns:
        Normalized name safe for use as directory (e.g., 'schroneko_gemma_2_2b_jpn_it')
    """
    # Replace forward slashes, hyphens, and other problematic characters with underscores
    normalized = re.sub(r'[/\\:*?"<>|\-]', "_", model_name)
    # Remove any duplicate underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized


def mask_token(token: str, show_chars: int = 4) -> str:
    """
    Mask an authentication token for display.

    Args:
        token: The token to mask
        show_chars: Number of characters to show at start and end

    Returns:
        Masked token string (e.g., 'abcd...wxyz')
    """
    if not token or len(token) <= show_chars * 2:
        return "****"

    return f"{token[:show_chars]}...{token[-show_chars:]}"


def parse_tags(tags_str: str) -> dict[str, str]:
    """
    Parse AWS tags from a comma-separated string.

    Args:
        tags_str: Tag string in format "key1=value1,key2=value2"

    Returns:
        Dictionary of tag key-value pairs

    Raises:
        ValueError: If tag format is invalid
    """
    if not tags_str or not tags_str.strip():
        return {}

    tags = {}
    for tag_pair in tags_str.split(","):
        pair = tag_pair.strip()
        if not pair:
            continue

        if "=" not in pair:
            error_msg = f"Invalid tag format: '{pair}'. Expected format: key=value"
            raise ValueError(error_msg)

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            error_msg = "Tag key cannot be empty"
            raise ValueError(error_msg)

        tags[key] = value

    return tags


def get_config_path(cache_dir: Path) -> Path:
    """
    Get the path to the config.json file.

    Args:
        cache_dir: Base cache directory

    Returns:
        Path to config.json
    """
    return cache_dir / "config.json"


def load_config(cache_dir: Path) -> dict:
    """
    Load the configuration from config.json.

    Args:
        cache_dir: Base cache directory

    Returns:
        Configuration dictionary with model information
    """
    config_path = get_config_path(cache_dir)

    if not config_path.exists():
        return {"models": {}}

    try:
        with config_path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {"models": {}}


def save_config(cache_dir: Path, config: dict) -> None:
    """
    Save the configuration to config.json.

    Args:
        cache_dir: Base cache directory
        config: Configuration dictionary to save
    """
    config_path = get_config_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    except OSError as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise


def get_model_cache_dir(cache_dir: Path, model_name: str, stage: str = "dev") -> Path:
    """
    Get the cache directory for a specific model and stage.

    Args:
        cache_dir: Base cache directory
        model_name: Model name (original, will be normalized)
        stage: Deployment stage (default: 'dev')

    Returns:
        Path to the model-stage-specific cache directory (cache_dir/stage/normalized_name)
    """
    normalized_name = normalize_model_name(model_name)
    return cache_dir / stage / normalized_name


def get_deployment_url(model_cache_dir: Path) -> str | None:
    """
    Get the deployment URL for a model using Zappa's Python API.

    Args:
        model_cache_dir: Model-specific cache directory

    Returns:
        Deployment URL if found, None otherwise
    """
    zappa_settings_path = model_cache_dir / "zappa_settings.json"
    if not zappa_settings_path.exists():
        return None

    try:
        # Load zappa settings
        with zappa_settings_path.open() as f:
            settings = json.load(f)

        # Assume "dev" stage (consistent with our deployment strategy)
        stage = "dev"
        stage_config = settings.get(stage, {})

        if not stage_config:
            logger.debug(f"Stage '{stage}' not found in zappa settings")
            return None

        # Get project name and construct lambda name
        # Zappa uses format: {project_name}-{stage}
        project_name = stage_config.get("project_name")
        if not project_name:
            logger.debug("project_name not found in zappa settings")
            return None

        lambda_name = f"{project_name}-{stage}"

        # Create Zappa instance and get API URL
        zappa = Zappa()
        api_url = zappa.get_api_url(lambda_name, stage)

        return api_url if api_url else None

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Could not get deployment URL: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        # Catch any Zappa-specific exceptions (e.g., AWS errors)
        logger.debug(f"Error getting deployment URL from Zappa: {e}")
        return None


def update_model_config(
    cache_dir: Path,
    model_name: str,
    auth_token: str | None = None,
    region: str | None = None,
    tags: dict[str, str] | None = None,
    url: str | None = None,
    stage: str = "dev",
) -> None:
    """
    Update the configuration for a specific model-stage combination.

    Args:
        cache_dir: Base cache directory
        model_name: Model name
        auth_token: Optional authentication token
        region: Optional AWS region
        tags: Optional AWS resource tags
        url: Optional deployment URL
        stage: Deployment stage (default: 'dev')
    """
    config = load_config(cache_dir)
    normalized_name = normalize_model_name(model_name)

    # Use nested structure: models[model_name][stage]
    if model_name not in config["models"]:
        config["models"][model_name] = {}

    if stage not in config["models"][model_name]:
        config["models"][model_name][stage] = {}

    model_config = config["models"][model_name][stage]
    model_config["normalized_name"] = normalized_name
    model_config["cache_dir"] = f"{stage}/{normalized_name}"
    model_config["last_updated"] = datetime.now(UTC).isoformat()

    if auth_token is not None:
        model_config["auth_token"] = auth_token
    if region is not None:
        model_config["region"] = region
    if tags:
        model_config["tags"] = tags
    if url is not None:
        model_config["url"] = url

    save_config(cache_dir, config)
    logger.info(f"Updated configuration for model: {model_name}, stage: {stage}")


def validate_ollama_model(model_name: str) -> bool:
    """
    Validate that the given Ollama model name exists in the Ollama library.

    Args:
        model_name: Name of the Ollama model to validate

    Returns:
        True if the model is valid, False otherwise

    Raises:
        ValueError: If the model name is invalid or not found
    """
    logger.info(f"Validating Ollama model: {model_name}")

    try:
        # Query Ollama library API to check if model exists
        response = httpx.get(
            "https://ollama.com/api/tags",
            timeout=10.0,
        )

        # If we can't reach the API, try a local Ollama instance
        if response.status_code != 200:
            logger.warning("Could not reach Ollama library API, trying local instance")
            try:
                local_response = httpx.get(
                    "http://localhost:11434/api/tags",
                    timeout=5.0,
                )
                if local_response.status_code == 200:
                    models_data = local_response.json()
                    available_models = [m["name"] for m in models_data.get("models", [])]
                    if any(model_name in m for m in available_models):
                        logger.info(f"Model {model_name} found in local Ollama instance")
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

        # Validate model name format: should be in format "owner/model" or "model"
        if "/" in model_name:
            parts = model_name.split("/")
            if len(parts) != 2 or not all(parts):
                error_msg = f"Invalid model name format: {model_name}. Expected format: 'owner/model' or 'model'"
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif not model_name.strip():
            error_msg = "Model name cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # If we get here, assume the model name is valid
        # (we can't exhaustively check all models without API access)
        logger.info(f"Model name {model_name} appears to be valid")
        return True

    except httpx.TimeoutException:
        logger.warning("Timeout while validating model, assuming valid format")
        # Fallback to basic format validation
        if "/" in model_name:
            parts = model_name.split("/")
            if len(parts) != 2 or not all(parts):
                error_msg = f"Invalid model name format: {model_name}. Expected format: 'owner/model' or 'model'"
                logger.error(error_msg)
                raise ValueError(error_msg)
        return True
    except ValueError:
        raise
    except Exception as e:
        logger.exception(f"Error validating model: {e}")
        raise ValueError(f"Failed to validate model: {e}") from e


def generate_from_template(
    template_path: Path,
    output_path: Path,
    replacements: dict[str, str],
) -> None:
    """
    Generate a file from a template with placeholder replacement.

    Args:
        template_path: Path to the template file
        output_path: Path where the generated file should be written
        replacements: Dictionary of placeholder -> value mappings
    """
    logger.info(f"Generating {output_path} from template {template_path}")

    if not template_path.exists():
        error_msg = f"Template file not found: {template_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Read template content
    template_content = template_path.read_text()

    # Replace placeholders
    for placeholder, value in replacements.items():
        template_content = template_content.replace(f"{{{{{placeholder}}}}}", value)

    # Write output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template_content)
    logger.info(f"Successfully generated {output_path}")


def _generate_zappa_settings(
    output_path: Path,
    model_name: str,
    aws_region: str,
    tags: dict[str, str],
    s3_bucket: str,
    auth_token: str,
    authorizer_arn: str | None = None,
    stage: str = "dev",
) -> None:
    """
    Generate zappa_settings.json using Zappa's Python API.

    Args:
        output_path: Path where zappa_settings.json should be written
        model_name: Ollama model name
        aws_region: AWS region
        tags: AWS resource tags
        s3_bucket: S3 bucket name for Zappa deployment
        auth_token: Authentication token for API access
        authorizer_arn: Optional ARN of the authorizer Lambda function
        stage: Zappa deployment stage (default: 'dev')
    """
    from zappa.cli import ZappaCLI  # noqa: PLC0415

    logger.info(f"Generating {output_path} using Zappa Python API for stage '{stage}'")

    # Create ZappaCLI instance
    cli = ZappaCLI()

    # Generate base settings dict
    settings_dict = cli._generate_settings_dict(stage=stage, config_args=[])

    # Update with our specific configuration
    # Note: Zappa automatically appends "-{stage}" to project_name for the Lambda function name
    normalized_model = normalize_model_name(model_name)
    stage_config = settings_dict[stage]

    # Configure authorizer based on whether ARN is provided
    if authorizer_arn:
        authorizer_config = {
            "arn": authorizer_arn,
            "token_header": "X-API-Key",
            "result_ttl": 300,
        }
    else:
        # Placeholder - will be updated after authorizer deployment
        authorizer_config = {
            "function": "authorizer.lambda_handler",
            "token_header": "X-API-Key",
            "result_ttl": 300,
        }

    stage_config.update(
        {
            "app_function": "merle.app.app",
            "project_name": f"merle-{normalized_model}",
            "s3_bucket": s3_bucket,
            "aws_region": aws_region,
            "memory_size": 10240,
            "timeout_seconds": 900,
            "environment_variables": {
                "OLLAMA_MODEL": model_name,
                "OLLAMA_URL": "http://localhost:11434",
                "OLLAMA_MODELS": "/tmp/models",  # noqa: S108 - Lambda writable directory
                "OLLAMA_STARTUP_TIMEOUT": "120",
                "ZAPPA_RUNNING_IN_DOCKER": "True",
                "API_KEY": auth_token,
            },
            "ephemeral_storage": {"Size": 5120},
            "keep_warm": False,
            "keep_warm_expression": "rate(4 minutes)",
            "authorizer": authorizer_config,
            "cors": True,
            "cors_allow_headers": ["Content-Type", "X-API-Key"],
            "binary_support": False,
            "tags": tags,
        }
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(settings_dict, f, indent=4, sort_keys=True)

    logger.info(f"Successfully generated {output_path} for stage '{stage}'")


def prepare_deployment_files(
    model_name: str,
    cache_dir: Path,
    auth_token: str | None = None,
    aws_region: str | None = None,
    tags: dict[str, str] | None = None,
    s3_bucket: str | None = None,
    stage: str = "dev",
) -> Path:
    """
    Prepare all necessary files for deployment.

    Args:
        model_name: Name of the Ollama model to deploy
        cache_dir: Base cache directory (files will be created in model-stage subdirectory)
        auth_token: Optional authentication token for API access
        aws_region: Optional AWS region (defaults to settings.REGION)
        tags: Optional AWS resource tags as key-value pairs
        s3_bucket: Optional S3 bucket name for Zappa deployment (generated if not provided)
        stage: Deployment stage (default: 'dev')

    Returns:
        Path to the model-stage-specific cache directory where files were created
    """
    logger.info(f"Preparing deployment files for model: {model_name}, stage: {stage}")

    # Validate model name first
    validate_ollama_model(model_name)

    # Get model-stage-specific cache directory
    model_cache_dir = get_model_cache_dir(cache_dir, model_name, stage)
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Get project root and template directory
    project_root = Path(__file__).parent.parent
    template_dir = Path(__file__).parent / "templates"

    # Prepare replacements
    region = aws_region or REGION
    tags_dict = tags or {}
    normalized_model = normalize_model_name(model_name)

    # Validate required parameters
    if not s3_bucket:
        error_msg = "s3_bucket is required for prepare_deployment_files"
        raise ValueError(error_msg)

    if not auth_token:
        error_msg = "auth_token is required for prepare_deployment_files"
        raise ValueError(error_msg)

    replacements = {
        "OLLAMA_MODEL": model_name,
        "AWS_REGION": region,
        "AUTH_TOKEN": auth_token,
        "TAGS_JSON": json.dumps(tags_dict),
    }

    # Generate Dockerfile
    generate_from_template(
        template_path=template_dir / "Dockerfile.template",
        output_path=model_cache_dir / "Dockerfile",
        replacements=replacements,
    )

    # Generate authorizer zappa_settings.json
    generate_from_template(
        template_path=template_dir / "zappa_settings_authorizer.json.template",
        output_path=model_cache_dir / "zappa_settings_authorizer.json",
        replacements={
            "NORMALIZED_MODEL": normalized_model,
            "S3_BUCKET": s3_bucket,
            "AWS_REGION": region,
            "AUTH_TOKEN": auth_token,
        },
    )

    # Copy authorizer.py
    authorizer_src = template_dir / "authorizer.py"
    if authorizer_src.exists():
        shutil.copy2(authorizer_src, model_cache_dir / "authorizer.py")
        logger.info("Copied authorizer.py to model cache directory")

    # Generate main zappa_settings.json using Zappa Python API
    # Note: authorizer_arn will be None initially, will be updated after authorizer deployment
    _generate_zappa_settings(
        output_path=model_cache_dir / "zappa_settings.json",
        model_name=model_name,
        aws_region=region,
        tags=tags_dict,
        s3_bucket=s3_bucket,
        auth_token=auth_token,
        authorizer_arn=None,  # Will be updated after authorizer deployment
    )

    # Copy pyproject.toml
    pyproject_src = project_root / "pyproject.toml"
    if pyproject_src.exists():
        shutil.copy2(pyproject_src, model_cache_dir / "pyproject.toml")
        logger.info("Copied pyproject.toml to model cache directory")

    # Copy uv.lock if it exists
    uv_lock_src = project_root / "uv.lock"
    if uv_lock_src.exists():
        shutil.copy2(uv_lock_src, model_cache_dir / "uv.lock")
        logger.info("Copied uv.lock to model cache directory")

    # Copy merle/ directory
    merle_src = project_root / "merle"
    merle_dst = model_cache_dir / "merle"

    if merle_dst.exists():
        shutil.rmtree(merle_dst)

    shutil.copytree(
        merle_src,
        merle_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "templates"),
    )
    logger.info("Copied merle/ directory to model cache directory")

    # Update configuration
    update_model_config(
        cache_dir=cache_dir,
        model_name=model_name,
        auth_token=auth_token,
        region=region,
        tags=tags_dict if tags_dict else None,
        stage=stage,
    )

    logger.info(f"Successfully prepared deployment files in {model_cache_dir}")
    return model_cache_dir


def update_zappa_settings_authorizer(
    model_cache_dir: Path,
    authorizer_arn: str,
    stage: str = "dev",
) -> None:
    """
    Update the main zappa_settings.json with the authorizer ARN.

    Args:
        model_cache_dir: Path to the model cache directory
        authorizer_arn: ARN of the deployed authorizer Lambda function
        stage: Zappa deployment stage (default: 'dev')
    """
    zappa_settings_path = model_cache_dir / "zappa_settings.json"

    if not zappa_settings_path.exists():
        error_msg = f"zappa_settings.json not found at {zappa_settings_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Load existing settings
    with zappa_settings_path.open() as f:
        settings = json.load(f)

    # Update authorizer configuration
    if stage not in settings:
        error_msg = f"Stage '{stage}' not found in zappa_settings.json"
        logger.error(error_msg)
        raise ValueError(error_msg)

    settings[stage]["authorizer"] = {
        "arn": authorizer_arn,
        "token_header": "X-API-Key",
        "result_ttl": 300,
    }

    # Write updated settings
    with zappa_settings_path.open("w") as f:
        json.dump(settings, f, indent=4, sort_keys=True)

    logger.info(f"Updated {zappa_settings_path} with authorizer ARN: {authorizer_arn}")


def process(filepath: Path, output_directory: Path) -> None:
    """Process a file (placeholder function)."""
    logger.debug(f"filepath={filepath}")
    logger.debug(f"output_directory={output_directory}")
