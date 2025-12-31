import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import boto3
import httpx
from zappa.cli import ZappaCLI
from zappa.core import Zappa

from merle.settings import REGION

logger = logging.getLogger(__name__)


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name to be safe for use as a directory name and ECR repository.

    Replaces special characters (including hyphens and periods) with underscores and
    lowercases the result to create a cross-platform safe directory name that's also
    valid for ECR and compatible with AWS resource naming conventions.

    Args:
        model_name: Original model name (e.g., 'hf.co/owner/model-name')

    Returns:
        Normalized name safe for use as directory (e.g., 'hf_co_owner_model_name')
    """
    # Replace forward slashes, hyphens, periods, and other problematic characters with underscores
    normalized = re.sub(r'[/\\:*?"<>|\-.]', "_", model_name)
    # Remove any duplicate underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    # Lowercase for ECR compatibility (ECR requires lowercase repository names)
    return normalized.lower()


def sanitize_for_cloudformation(name: str) -> str:
    """
    Sanitize a name for use as a CloudFormation stack name and Lambda function name.

    CloudFormation stack names must match: [a-zA-Z][-a-zA-Z0-9]*
    Lambda function names can only contain: a-z, A-Z, 0-9, _ and -

    Args:
        name: Input name (may contain underscores or other invalid characters)

    Returns:
        Sanitized name safe for CloudFormation and Lambda (invalid chars replaced with hyphens)
    """
    # Replace underscores, periods, and other invalid characters with hyphens
    sanitized = re.sub(r"[_./\\:*?\"<>|]", "-", name)
    # Remove any duplicate hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Ensure starts with a letter (prepend 'p-' if it starts with a number)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"p-{sanitized}"
    # Lowercase for consistency with ECR and other AWS resource naming
    return sanitized.lower()


def get_default_project_name() -> str:
    """
    Get the default project name from the current working directory name.

    Returns:
        Normalized parent directory name as the default project name
    """
    cwd_name = Path.cwd().name
    return normalize_model_name(cwd_name)


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


def read_system_prompt(system_prompt_arg: str | None) -> str | None:
    """
    Read system prompt from string or file.

    Args:
        system_prompt_arg: Either a system prompt string or path to a UTF-8 text file

    Returns:
        System prompt string or None if not provided

    Raises:
        ValueError: If file path is provided but file cannot be read
    """
    if not system_prompt_arg:
        return None

    # Check if it looks like a file path
    potential_path = Path(system_prompt_arg)
    if potential_path.exists() and potential_path.is_file():
        try:
            system_prompt = potential_path.read_text(encoding="utf-8")
            logger.info(f"Loaded system prompt from file: {potential_path}")
            return system_prompt.strip()
        except (OSError, UnicodeDecodeError) as e:
            error_msg = f"Failed to read system prompt from file {potential_path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    # Treat as direct string
    logger.info("Using system prompt from argument")
    return system_prompt_arg.strip()


def get_model_context_window_size(model_name: str) -> int:
    """
    Get the default context window size for an Ollama model.

    Returns a default context window size based on known model families.
    The actual size can be configured via num_ctx parameter when running the model.

    Args:
        model_name: Name of the model (e.g., 'llama2', 'mistral', 'tinyllama')

    Returns:
        Default context window size in tokens (defaults to 2048 if unknown)
    """
    # Extract base model name (handle variants like llama2:7b, mistral:latest, etc.)
    base_name = model_name.split(":")[0].lower()

    # Known model context window sizes (defaults before num_ctx override)
    model_defaults = {
        # LLaMA family
        "llama2": 4096,
        "llama3": 8192,
        "llama3.1": 131072,  # 128K context
        "llama3.2": 131072,
        "codellama": 16384,
        # Mistral family
        "mistral": 8192,
        "mixtral": 32768,
        # Gemma family
        "gemma": 8192,
        "gemma2": 8192,
        # Other popular models
        "phi": 2048,
        "phi3": 4096,
        "tinyllama": 2048,
        "qwen": 8192,
        "deepseek": 4096,
        "yi": 4096,
        "solar": 4096,
        "dolphin": 4096,
        "orca": 4096,
        "vicuna": 2048,
        "starling": 8192,
        "neural": 4096,
        "openchat": 8192,
    }

    # Check if base model name matches any known models
    for known_model, context_size in model_defaults.items():
        if base_name.startswith(known_model):
            logger.info(f"Using default context window size for {model_name}: {context_size} tokens")
            return context_size

    # Default fallback
    default_size = 2048
    logger.info(
        f"Unknown model {model_name}, using default context window size: {default_size} tokens. "
        f"This can be overridden via OLLAMA_CONTEXT_LENGTH environment variable."
    )
    return default_size


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


def get_project_cache_dir(cache_dir: Path, project_name: str | None = None) -> Path:
    """
    Get the project-specific cache directory.

    Prefixes the cache directory by the normalized project name to support
    multiple projects using the same base cache directory.

    Args:
        cache_dir: Base cache directory
        project_name: Project name (optional, will be normalized if provided)

    Returns:
        Path to the project-specific cache directory (cache_dir/normalized_project_name)
        If project_name is None, returns cache_dir unchanged.
    """
    if project_name is None:
        return cache_dir
    normalized_project = normalize_model_name(project_name)
    return cache_dir / normalized_project


def get_model_cache_dir(cache_dir: Path, model_name: str, stage: str = "dev") -> Path:
    """
    Get the cache directory for a specific model and stage.

    Args:
        cache_dir: Base cache directory (should already be project-specific if using projects)
        model_name: Model name (original, will be normalized)
        stage: Deployment stage (default: 'dev')

    Returns:
        Path to the model-stage-specific cache directory (cache_dir/stage/normalized_name)
    """
    normalized_name = normalize_model_name(model_name)
    return cache_dir / stage / normalized_name


def get_deployment_url(model_cache_dir: Path) -> str | None:  # noqa: PLR0911
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
        aws_region = stage_config.get("aws_region", "us-east-1")

        # Check if CloudFormation stack exists before calling Zappa
        # This avoids Zappa logging ERROR when stack doesn't exist
        cf_client = boto3.client("cloudformation", region_name=aws_region)
        try:
            cf_client.describe_stacks(StackName=lambda_name)
        except cf_client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ValidationError":
                # Stack doesn't exist - model is configured but not deployed
                logger.debug(f"CloudFormation stack '{lambda_name}' does not exist")
                return None
            raise

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


def update_model_config(  # noqa: C901, PLR0912
    cache_dir: Path,
    model_name: str,
    auth_token: str | None = None,
    region: str | None = None,
    tags: dict[str, str] | None = None,
    url: str | None = None,
    stage: str = "dev",
    system_prompt: str | None = None,
    context_window_size: int | None = None,
    use_split: bool | None = None,
    split_config: dict | None = None,
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
        system_prompt: Optional system prompt for chat context
        context_window_size: Optional context window size in tokens
        use_split: Optional flag indicating if split model mode is used
        split_config: Optional split model configuration dict
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
    if system_prompt is not None:
        model_config["system_prompt"] = system_prompt
    if context_window_size is not None:
        model_config["context_window_size"] = context_window_size
    if use_split is not None:
        model_config["use_split"] = use_split
    if split_config is not None:
        model_config["split"] = split_config

    save_config(cache_dir, config)
    logger.info(f"Updated configuration for model: {model_name}, stage: {stage}")


def _validate_model_name_format(model_name: str) -> None:
    """
    Validate the format of an Ollama model name.

    Supports formats: "model", "owner/model", "hf.co/owner/model"

    Args:
        model_name: Name of the Ollama model to validate

    Raises:
        ValueError: If the model name format is invalid
    """
    if not model_name.strip():
        error_msg = "Model name cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if "/" not in model_name:
        return  # Simple model name like "llama2" is valid

    parts = model_name.split("/")
    is_hf_model = model_name.startswith("hf.co/")

    if is_hf_model:
        if len(parts) != 3 or not all(parts):
            error_msg = f"Invalid HuggingFace model format: {model_name}. Expected: 'hf.co/owner/model'"
            logger.error(error_msg)
            raise ValueError(error_msg)
    elif len(parts) != 2 or not all(parts):
        error_msg = f"Invalid model name format: {model_name}. Expected: 'owner/model' or 'model'"
        logger.error(error_msg)
        raise ValueError(error_msg)


def _check_local_ollama(model_name: str) -> bool:
    """
    Check if a model exists in the local Ollama instance.

    Args:
        model_name: Name of the model to check

    Returns:
        True if found locally, False otherwise
    """
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
        logger.debug("Local Ollama instance not available")
    return False


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
        # Validate model name format first
        # Supports: "model", "owner/model", "hf.co/owner/model"
        _validate_model_name_format(model_name)

        # Check local Ollama instance for custom/HuggingFace models
        if _check_local_ollama(model_name):
            return True

        # If we get here, assume the model name is valid
        logger.info(f"Model name {model_name} appears to be valid")
        return True

    except httpx.TimeoutException:
        logger.warning("Timeout while validating model, assuming valid format")
        _validate_model_name_format(model_name)
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
    project_name: str,
    stage: str = "dev",
    memory_size: int = 8192,
    context_window_size: int | None = None,
    use_split: bool = False,
    ephemeral_storage: int = 5120,
) -> None:
    """
    Generate zappa_settings.json using Zappa's Python API.

    Uses embedded authorizer (authorizer.lambda_handler) for API authentication.

    Args:
        output_path: Path where zappa_settings.json should be written
        model_name: Ollama model name
        aws_region: AWS region
        tags: AWS resource tags
        s3_bucket: S3 bucket name for Zappa deployment
        auth_token: Authentication token for API access
        project_name: Project name for the Lambda function (used as prefix)
        stage: Zappa deployment stage (default: 'dev')
        memory_size: Lambda function memory size in MB (default: 8192)
        context_window_size: Optional context window size in tokens
        use_split: Whether to use split model mode (default: False)
        ephemeral_storage: Ephemeral storage in MB (default: 5120)
    """
    mode_str = " (split mode)" if use_split else ""
    logger.info(f"Generating {output_path} using Zappa Python API for stage '{stage}'{mode_str}")

    # Create ZappaCLI instance
    cli = ZappaCLI()

    # Generate base settings dict
    settings_dict = cli._generate_settings_dict(stage=stage, config_args=[])

    # Update with our specific configuration
    # Note: Zappa automatically appends "-{stage}" to project_name for the Lambda function name
    # CloudFormation stack names must match [a-zA-Z][-a-zA-Z0-9]* (no underscores allowed)
    sanitized_project = sanitize_for_cloudformation(project_name)
    stage_config = settings_dict[stage]

    # Configure embedded authorizer for API authentication
    authorizer_config = {
        "function": "authorizer.lambda_handler",
        "token_header": "X-API-Key",
        "result_ttl": 300,
    }

    # Build environment variables
    # Models are in Docker image and copied/reassembled to /tmp/models at runtime
    models_path = "/tmp/models"  # noqa: S108
    env_vars = {
        "OLLAMA_MODEL": model_name,
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODELS": models_path,
        "OLLAMA_STARTUP_TIMEOUT": "120",
        "ZAPPA_RUNNING_IN_DOCKER": "True",
        "API_KEY": auth_token,
    }

    # Add context window size if provided
    if context_window_size:
        env_vars["OLLAMA_MODEL_CONTEXT_WINDOW_SIZE"] = str(context_window_size)

    # Add split model indicator
    if use_split:
        env_vars["MERLE_SPLIT_MODEL"] = "true"
        # Increase startup timeout for S3 download
        env_vars["OLLAMA_STARTUP_TIMEOUT"] = "300"

    stage_config.update(
        {
            "app_function": "merle.app.app",
            "project_name": sanitized_project,
            "s3_bucket": s3_bucket,
            "aws_region": aws_region,
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

    # Add S3 permissions for split model mode
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(settings_dict, f, indent=4, sort_keys=True)

    logger.info(f"Successfully generated {output_path} for stage '{stage}'")


def prepare_deployment_files(  # noqa: PLR0915, PLR0912
    model_name: str,
    cache_dir: Path,
    project_name: str,
    auth_token: str | None = None,
    aws_region: str | None = None,
    tags: dict[str, str] | None = None,
    s3_bucket: str | None = None,
    stage: str = "dev",
    memory_size: int = 8192,
    system_prompt: str | None = None,
    skip_model_download: bool = False,
) -> Path:
    """
    Prepare all necessary files for deployment.

    This function automatically handles model size detection and splitting:
    - Downloads the model using local Ollama
    - Calculates if the model fits in a Docker image (~8GB limit)
    - If too large, splits the model and uploads overflow to S3
    - Generates appropriate Dockerfile (standard or split mode)

    Args:
        model_name: Name of the Ollama model to deploy
        cache_dir: Base cache directory (files will be created in model-stage subdirectory)
        project_name: Project name for the Lambda function (used as prefix in zappa project_name)
        auth_token: Optional authentication token for API access
        aws_region: Optional AWS region (defaults to settings.REGION)
        tags: Optional AWS resource tags as key-value pairs
        s3_bucket: Optional S3 bucket name for Zappa deployment (generated if not provided)
        stage: Deployment stage (default: 'dev')
        memory_size: Lambda function memory size in MB (default: 8192)
        system_prompt: Optional system prompt for chat context
        skip_model_download: Skip model download (for testing, default: False)

    Returns:
        Path to the model-stage-specific cache directory where files were created
    """
    logger.info(f"Preparing deployment files for model: {model_name}, stage: {stage}")

    # Validate model name first
    validate_ollama_model(model_name)

    # Get model-stage-specific cache directory
    model_cache_dir = get_model_cache_dir(cache_dir, model_name, stage)
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Get template directory from installed package
    template_dir = Path(__file__).parent / "templates"

    # Get consuming project root from current working directory
    # (where the CLI is being run, not the installed package location)
    consuming_project_root = Path.cwd()

    # Prepare replacements
    region = aws_region or REGION
    tags_dict = tags or {}

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

    # Determine if we need to download and potentially split the model
    use_split = False
    split_metadata = None
    size_details: dict | None = None

    if not skip_model_download:
        # Import model splitting module (here to avoid circular imports at module load)
        from merle.model_split import (  # noqa: PLC0415
            calculate_model_size,
            copy_model_to_output,
            download_model_locally,
            needs_splitting,
            prepare_split_model,
        )

        # Download the model using local Ollama
        logger.info(f"Downloading model {model_name} using local Ollama...")
        try:
            download_model_locally(model_name)
        except RuntimeError as e:
            logger.error(f"Failed to download model: {e}")
            raise ValueError(f"Failed to download model: {e}") from e

        # Calculate model size
        total_size, size_details = calculate_model_size(model_name)
        logger.info(f"Model size: {size_details['total_size_gb']} GB")

        # Check if splitting is needed
        if needs_splitting(total_size):
            use_split = True
            logger.info("Model exceeds Docker image limit, preparing split deployment...")

            # Prepare split model (downloads to S3, creates part files)
            split_metadata = prepare_split_model(
                model_name=model_name,
                output_dir=model_cache_dir,
                s3_bucket=s3_bucket,
                region=region,
            )

            logger.info("Split model prepared:")
            logger.info(f"  - Image portion: {split_metadata['image_portion_bytes'] / (1024**3):.2f} GB")
            logger.info(f"  - S3 portion: {split_metadata['s3_portion_bytes'] / (1024**3):.2f} GB")
            logger.info(f"  - S3 URI: {split_metadata['s3']['uri']}")
        else:
            logger.info("Model fits in Docker image, using standard deployment")
            # Copy model files to output directory for Docker image inclusion
            copy_model_to_output(model_name, model_cache_dir)

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
        output_path=model_cache_dir / "Dockerfile",
        replacements=replacements,
    )

    # Copy authorizer.py
    authorizer_src = template_dir / "authorizer.py"
    if authorizer_src.exists():
        shutil.copy2(authorizer_src, model_cache_dir / "authorizer.py")
        logger.info("Copied authorizer.py to model cache directory")

    # Get context window size for the model
    context_window_size = get_model_context_window_size(model_name)

    # Calculate ephemeral storage needed based on model size
    # Lambda ephemeral storage (/tmp) ranges from 512 MB to 10,240 MB
    if size_details:
        model_size_gb = size_details["total_size_gb"]
        # Add 20% buffer and round up to nearest 512MB
        needed_mb = int((model_size_gb * 1024 * 1.2 + 511) // 512 * 512)
        # Clamp to Lambda limits: min 512 MB, max 10,240 MB
        ephemeral_storage = min(max(needed_mb, 512), 10240)
        logger.info(f"Setting ephemeral storage to {ephemeral_storage} MB for {model_size_gb:.2f} GB model")
    else:
        # Default when skip_model_download=True (model size unknown)
        ephemeral_storage = 5120  # 5GB default
        logger.info(f"Setting ephemeral storage to {ephemeral_storage} MB (default, model size unknown)")

    # Generate main zappa_settings.json using Zappa Python API
    # Uses embedded authorizer (authorizer.lambda_handler function in same Lambda)
    _generate_zappa_settings(
        output_path=model_cache_dir / "zappa_settings.json",
        model_name=model_name,
        aws_region=region,
        tags=tags_dict,
        s3_bucket=s3_bucket,
        auth_token=auth_token,
        project_name=project_name,
        memory_size=memory_size,
        context_window_size=context_window_size,
        use_split=use_split,
        ephemeral_storage=ephemeral_storage,
    )

    # Copy pyproject.toml from consuming project (where CLI is run)
    pyproject_src = consuming_project_root / "pyproject.toml"
    if pyproject_src.exists():
        shutil.copy2(pyproject_src, model_cache_dir / "pyproject.toml")
        logger.info(f"Copied pyproject.toml from consuming project: {consuming_project_root}")
    else:
        logger.warning(f"pyproject.toml not found in consuming project: {consuming_project_root}")

    # Note: We intentionally do NOT copy uv.lock or merle/ because:
    # 1. The Docker image may use a different architecture than the host
    # 2. uv will fetch dependencies (including merle from git) during Docker build
    # This ensures the correct platform-specific packages are installed.

    # Update configuration
    update_model_config(
        cache_dir=cache_dir,
        model_name=model_name,
        auth_token=auth_token,
        region=region,
        tags=tags_dict if tags_dict else None,
        stage=stage,
        system_prompt=system_prompt,
        context_window_size=context_window_size,
        use_split=use_split,
        split_config=split_metadata,
    )

    logger.info(f"Successfully prepared deployment files in {model_cache_dir}")
    return model_cache_dir
