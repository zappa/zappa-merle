"""CLI for managing Ollama model deployment to AWS Lambda."""

import argparse
import base64
import json
import logging
import secrets
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from merle import __version__
from merle.chat import run_interactive_chat
from merle.functions import (
    get_default_project_name,
    get_deployment_url,
    get_model_cache_dir,
    get_project_cache_dir,
    load_config,
    mask_token,
    parse_tags,
    prepare_deployment_files,
    read_system_prompt,
    save_config,
)
from merle.settings import (
    LAMBDA_MEMORY_SIZE_DEFAULT,
    LAMBDA_MEMORY_SIZE_MAX,
    LAMBDA_MEMORY_SIZE_MIN,
    REGION,
    STAGE,
    validate_lambda_memory_size,
)

logger = logging.getLogger(__name__)

# Display constants
MAX_MODEL_NAME_LENGTH = 30
MAX_SYSTEM_PROMPT_DISPLAY_LENGTH = 40


def generate_unique_bucket_name() -> str:
    """
    Generate a unique S3 bucket name using UUID.

    Returns:
        Unique bucket name in format: zappa-merle-<short-uuid>
    """
    # Generate a short UUID (first 8 characters)
    short_uuid = str(uuid.uuid4())[:8]
    return f"zappa-merle-{short_uuid}"


def get_config_directory() -> Path:
    """
    Get the merle configuration directory.

    Returns cross-platform user configuration directory (~/.merle).

    Returns:
        Path to the configuration directory
    """
    config_dir = Path.home() / ".merle"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_effective_project_name(args: argparse.Namespace) -> str:
    """
    Get the effective project name based on args.

    Uses --project if specified, otherwise defaults to the current directory name.

    Args:
        args: Parsed command-line arguments (expects project attribute)

    Returns:
        Project name (from --project or default from current directory name)
    """
    project_name = getattr(args, "project", None)
    if project_name is None:
        project_name = get_default_project_name()
        logger.info(f"Using default project name from current directory: {project_name}")
    return project_name


def get_effective_cache_dir(args: argparse.Namespace) -> Path:
    """
    Get the effective cache directory based on args.

    Applies project prefix (from --project or defaulting to current directory name).

    Args:
        args: Parsed command-line arguments (expects cache_dir and project attributes)

    Returns:
        Path to the effective cache directory (with project prefix)
    """
    # Determine base cache directory
    if hasattr(args, "cache_dir") and args.cache_dir is not None:
        base_cache_dir = Path(args.cache_dir).resolve()
    else:
        base_cache_dir = get_config_directory()

    # Get project name (from --project or default)
    project_name = get_effective_project_name(args)

    cache_dir = get_project_cache_dir(base_cache_dir, project_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_model_to_use(cache_dir: Path, specified_model: str | None) -> str:
    """
    Determine which model to use based on config and user input.

    Args:
        cache_dir: Base cache directory
        specified_model: Model specified by user (or None)

    Returns:
        Model name to use

    Raises:
        ValueError: If model cannot be determined
    """
    if specified_model:
        return specified_model

    # Load config to check available models
    config = load_config(cache_dir)
    models = list(config.get("models", {}).keys())

    if len(models) == 0:
        error_msg = "No models configured. Please specify --model."
        raise ValueError(error_msg)
    if len(models) == 1:
        logger.info(f"Using the only configured model: {models[0]}")
        return models[0]

    # Multiple models exist
    error_msg = (
        f"Multiple models are deployed ({len(models)} models: {', '.join(models)}). "
        f"Please specify which model to use with --model."
    )
    raise ValueError(error_msg)


def handle_prepare_dockerfile(args: argparse.Namespace) -> int:  # noqa: C901, PLR0912
    """
    Handle the prepare-dockerfile command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get effective cache directory (with project prefix if specified)
        cache_dir = get_effective_cache_dir(args)

        stage = args.stage
        project_info = f", project: {args.project}" if getattr(args, "project", None) else ""
        logger.info(f"Preparing deployment files for model: {args.model}, stage: {stage}{project_info}")
        logger.info(f"Cache directory: {cache_dir}")

        # Validate memory size
        try:
            validate_lambda_memory_size(args.memory_size)
            logger.info(f"Lambda memory size: {args.memory_size} MB")
        except ValueError as e:
            logger.error(f"Invalid memory size: {e}")
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Parse tags if provided
        tags = None
        if args.tags:
            try:
                tags = parse_tags(args.tags)
                logger.info(f"Parsed tags: {tags}")
            except ValueError as e:
                logger.error(f"Invalid tags format: {e}")
                print(f"Error: {e}", file=sys.stderr)
                return 1

        # Read system prompt if provided
        system_prompt = None
        if hasattr(args, "system_prompt") and args.system_prompt:
            try:
                system_prompt = read_system_prompt(args.system_prompt)
                if system_prompt:
                    logger.info(f"System prompt configured ({len(system_prompt)} characters)")
            except ValueError as e:
                logger.error(f"Invalid system prompt: {e}")
                print(f"Error: {e}", file=sys.stderr)
                return 1

        # Load existing configuration to check for existing values
        # Use nested structure: models[model_name][stage]
        config = load_config(cache_dir)
        existing_model_config = config.get("models", {}).get(args.model, {}).get(stage, {})
        existing_auth_token = existing_model_config.get("auth_token")

        # Check if zappa_settings.json already exists
        model_cache_dir_check = get_model_cache_dir(cache_dir, args.model, stage)
        zappa_settings_path = model_cache_dir_check / "zappa_settings.json"

        existing_s3_bucket = None
        if zappa_settings_path.exists():
            # Load existing settings to get the S3 bucket
            try:
                with zappa_settings_path.open() as f:
                    existing_settings = json.load(f)
                    existing_s3_bucket = existing_settings.get("dev", {}).get("s3_bucket")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read existing zappa_settings.json: {e}")

        # Determine S3 bucket name
        if existing_s3_bucket:
            # Settings already exist - bucket cannot be changed
            if args.s3_bucket and args.s3_bucket != existing_s3_bucket:
                error_msg = (
                    f"Cannot change S3 bucket for existing deployment.\n"
                    f"Current bucket: {existing_s3_bucket}\n"
                    f"Requested bucket: {args.s3_bucket}\n"
                    f"To use a different bucket, first destroy the deployment with: merle destroy --model {args.model}"
                )
                logger.error(error_msg)
                print(f"Error: {error_msg}", file=sys.stderr)
                return 1
            s3_bucket = existing_s3_bucket
            logger.info(f"Using existing S3 bucket: {s3_bucket}")
        else:
            # No existing settings - use provided or generate new
            s3_bucket = (
                args.s3_bucket if hasattr(args, "s3_bucket") and args.s3_bucket else generate_unique_bucket_name()
            )
            logger.info(f"Using S3 bucket: {s3_bucket}")

        # Determine auth token (immutable once set)
        auth_token_is_new = False
        if existing_auth_token:
            # Token already exists - cannot be changed
            if args.auth_token and args.auth_token != existing_auth_token:
                error_msg = (
                    f"Cannot change auth token for existing model.\n"
                    f"To use a different token, first destroy the deployment with: merle destroy --model {args.model}"
                )
                logger.error(error_msg)
                print(f"Error: {error_msg}", file=sys.stderr)
                return 1
            auth_token = existing_auth_token
            logger.info("Using existing auth token")
        else:
            # No existing token - use provided or generate new (256 bits of randomness)
            auth_token = args.auth_token if args.auth_token else secrets.token_urlsafe(32)
            auth_token_is_new = not args.auth_token
            if auth_token_is_new:
                logger.info("Generated new secure authentication token")

        model_cache_dir = prepare_deployment_files(
            model_name=args.model,
            cache_dir=cache_dir,
            project_name=get_effective_project_name(args),
            auth_token=auth_token,
            aws_region=args.region,
            tags=tags,
            s3_bucket=s3_bucket,
            stage=stage,
            memory_size=args.memory_size,
            system_prompt=system_prompt,
        )

        # Display the auth token if it was newly generated
        if auth_token_is_new:
            print(f"\n{'=' * 80}")
            print("IMPORTANT: Generated new authentication token")
            print(f"{'=' * 80}")
            print(f"Token: {auth_token}")
            print(f"{'=' * 80}")
            print("Save this token securely - you'll need it to access your deployed API.")
            print(f"{'=' * 80}\n")

        print(f"Successfully prepared deployment files in: {model_cache_dir}")
        print("\nGenerated files:")
        print(f"  - {model_cache_dir / 'Dockerfile'}")
        print(f"  - {model_cache_dir / 'zappa_settings.json'}")
        print(f"  - {model_cache_dir / 'authorizer.py'}")
        print(f"\nConfiguration updated in: {cache_dir / 'config.json'}")
        print("\nNext steps:")
        print("  1. Review the generated files")
        print("  2. Build the Docker image")
        print(f"  3. Run: python -m merle.cli deploy --model {args.model} --auth-token YOUR_TOKEN")

        return 0

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"Error preparing deployment files: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_deploy(args: argparse.Namespace) -> int:  # noqa: C901, PLR0912
    """
    Handle the deploy command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        logger.info("Starting deployment using Zappa")

        # Get effective cache directory (with project prefix if specified)
        cache_dir = get_effective_cache_dir(args)

        stage = args.stage
        project_info = f", project: {args.project}" if getattr(args, "project", None) else ""
        logger.info(f"stage: {stage}{project_info}")

        # Determine which model to use
        try:
            model_name = get_model_to_use(cache_dir, args.model)
        except ValueError as e:
            logger.error(str(e))
            print(f"Error: {e}", file=sys.stderr)
            return 1

        logger.info(f"Deploying model: {model_name}")

        # Get model-stage-specific cache directory
        model_cache_dir = get_model_cache_dir(cache_dir, model_name, stage)
        zappa_settings = model_cache_dir / "zappa_settings.json"

        # Check if preparation is needed
        if not zappa_settings.exists():
            logger.info(f"Deployment files not found. Preparing model: {model_name}")
            print(f"Preparing deployment files for model: {model_name}...")

            # Validate memory size
            try:
                validate_lambda_memory_size(args.memory_size)
                logger.info(f"Lambda memory size: {args.memory_size} MB")
            except ValueError as e:
                logger.error(f"Invalid memory size: {e}")
                print(f"Error: {e}", file=sys.stderr)
                return 1

            # Parse tags if provided
            tags = None
            if args.tags:
                try:
                    tags = parse_tags(args.tags)
                    logger.info(f"Parsed tags: {tags}")
                except ValueError as e:
                    logger.error(f"Invalid tags format: {e}")
                    print(f"Error: {e}", file=sys.stderr)
                    return 1

            # Read system prompt if provided
            system_prompt = None
            if hasattr(args, "system_prompt") and args.system_prompt:
                try:
                    system_prompt = read_system_prompt(args.system_prompt)
                    if system_prompt:
                        logger.info(f"System prompt configured ({len(system_prompt)} characters)")
                except ValueError as e:
                    logger.error(f"Invalid system prompt: {e}")
                    print(f"Error: {e}", file=sys.stderr)
                    return 1

            # Determine S3 bucket name
            s3_bucket = (
                args.s3_bucket if hasattr(args, "s3_bucket") and args.s3_bucket else generate_unique_bucket_name()
            )
            logger.info(f"Using S3 bucket: {s3_bucket}")

            # Generate auth token if not provided (256 bits of randomness)
            auth_token = args.auth_token if args.auth_token else secrets.token_urlsafe(32)
            auth_token_is_new = not args.auth_token
            if auth_token_is_new:
                logger.info("Generated new secure authentication token")

            try:
                model_cache_dir = prepare_deployment_files(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    project_name=get_effective_project_name(args),
                    auth_token=auth_token,
                    aws_region=args.region,
                    tags=tags,
                    s3_bucket=s3_bucket,
                    stage=stage,
                    memory_size=args.memory_size,
                    system_prompt=system_prompt,
                )

                # Display the auth token if it was newly generated
                if auth_token_is_new:
                    print(f"\n{'=' * 80}")
                    print("IMPORTANT: Generated new authentication token")
                    print(f"{'=' * 80}")
                    print(f"Token: {auth_token}")
                    print(f"{'=' * 80}")
                    print("Save this token securely - you'll need it to access your deployed API.")
                    print(f"{'=' * 80}\n")
                print(f"Deployment files prepared in: {model_cache_dir}")
            except ValueError as e:
                logger.error(f"Failed to prepare deployment files: {e}")
                print(f"Error: {e}", file=sys.stderr)
                return 1

        logger.info(f"Using zappa_settings.json from: {zappa_settings}")
        logger.info(f"Working directory: {model_cache_dir}")

        # Determine auth token - use provided token or get from config
        auth_token = args.auth_token
        if not auth_token:
            config = load_config(cache_dir)
            model_config = config.get("models", {}).get(model_name, {}).get(stage, {})
            auth_token = model_config.get("auth_token")
            if auth_token:
                logger.info("Using auth token from configuration")
            else:
                error_msg = (
                    f"No auth token found for model: {model_name}, stage: {stage}. "
                    f"Please provide --auth-token or run 'merle prepare --model {model_name} --stage {stage}' first."
                )
                logger.error(error_msg)
                print(f"Error: {error_msg}", file=sys.stderr)
                return 1

        # Set auth token in environment
        import os

        env = os.environ.copy()
        env["API_KEY"] = auth_token
        logger.info("Auth token set in environment")

        # Build and push Docker image to ECR
        print("\n" + "=" * 80)
        print("Building and pushing Docker image to ECR")
        print("=" * 80)

        try:
            # Get AWS region from zappa settings
            with zappa_settings.open() as f:
                settings_data = json.load(f)
            aws_region = settings_data.get("dev", {}).get("aws_region", REGION)

            # Create ECR repository and build/push image
            import boto3

            from merle.functions import normalize_model_name

            normalized_name = normalize_model_name(model_name)
            ecr_repo_name = f"merle-{normalized_name}"

            # Create ECR client
            ecr_client = boto3.client("ecr", region_name=aws_region)

            # Create repository if it doesn't exist
            try:
                logger.info(f"Creating ECR repository: {ecr_repo_name}")
                ecr_client.create_repository(
                    repositoryName=ecr_repo_name,
                    imageScanningConfiguration={"scanOnPush": True},
                )
                print(f"✓ Created ECR repository: {ecr_repo_name}")
            except ecr_client.exceptions.RepositoryAlreadyExistsException:
                logger.info(f"ECR repository already exists: {ecr_repo_name}")
                print(f"✓ ECR repository already exists: {ecr_repo_name}")

            # Get repository URI
            response = ecr_client.describe_repositories(repositoryNames=[ecr_repo_name])
            repo_uri = response["repositories"][0]["repositoryUri"]
            image_uri = f"{repo_uri}:latest"
            logger.info(f"ECR image URI: {image_uri}")

            # Authenticate Docker to ECR using boto3 (works better with MFA-cached credentials)
            print("Authenticating with ECR...")
            auth_response = ecr_client.get_authorization_token()
            auth_token_ecr = auth_response["authorizationData"][0]["authorizationToken"]
            ecr_endpoint = auth_response["authorizationData"][0]["proxyEndpoint"]

            # Decode auth token (it's base64 encoded "AWS:password")
            username, password = base64.b64decode(auth_token_ecr).decode().split(":")

            # Docker login to ECR
            docker_login_cmd = ["docker", "login", "--username", username, "--password-stdin", ecr_endpoint]
            subprocess.run(
                docker_login_cmd,
                input=password.encode(),
                check=True,
                capture_output=True,
            )
            print("✓ Authenticated with ECR")

            # Build Docker image
            print(f"Building Docker image: {image_uri}")
            docker_build_cmd = ["docker", "build", "-t", image_uri, "."]
            logger.info(f"Running: {' '.join(docker_build_cmd)}")
            result = subprocess.run(
                docker_build_cmd,
                cwd=model_cache_dir,
                check=True,
                capture_output=False,
            )
            print(f"✓ Built Docker image: {image_uri}")

            # Push Docker image to ECR
            print("Pushing Docker image to ECR...")
            docker_push_cmd = ["docker", "push", image_uri]
            logger.info(f"Running: {' '.join(docker_push_cmd)}")
            result = subprocess.run(
                docker_push_cmd,
                cwd=model_cache_dir,
                check=True,
                capture_output=False,
            )
            print(f"✓ Pushed Docker image to ECR: {image_uri}")

            # Update zappa_settings.json with docker_image_uri
            settings_data["dev"]["docker_image_uri"] = image_uri
            with zappa_settings.open("w") as f:
                json.dump(settings_data, f, indent=4, sort_keys=True)
            logger.info(f"Updated zappa_settings.json with docker_image_uri: {image_uri}")
            print("✓ Updated zappa_settings.json with docker_image_uri")

        except subprocess.CalledProcessError as e:
            error_msg = f"Docker command failed: {e}"
            logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            error_msg = f"Failed to build/push Docker image: {e}"
            logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return 1

        print("=" * 80 + "\n")

        # Run zappa deploy from the model cache directory with Docker image URI
        cmd = ["zappa", "deploy", "dev", "--docker-image-uri", image_uri]
        logger.info(f"Running: {' '.join(cmd)}")
        print(f"Deploying with Zappa: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            env=env,
            cwd=model_cache_dir,
            check=False,
            capture_output=False,
        )

        if result.returncode == 0:
            # Get deployment URL and save to config
            deployment_url = get_deployment_url(model_cache_dir)
            if deployment_url:
                from merle.functions import update_model_config

                update_model_config(
                    cache_dir=cache_dir,
                    model_name=model_name,
                    url=deployment_url,
                    stage=stage,
                )
                logger.info(f"Saved deployment URL to config: {deployment_url}")

            print("\nDeployment successful!")
            print("Your Ollama model server is now available on AWS Lambda.")
            if deployment_url:
                print(f"\nDeployment URL: {deployment_url}")
            print("\nNext steps:")
            print("  # List deployed models")
            print("  uvx merle list")
            print("\n  # Start an interactive chat session")
            print(f"  uvx merle chat --model {model_name}")
            return 0

        logger.error(f"Deployment failed with exit code: {result.returncode}")
        print(f"\nDeployment failed with exit code: {result.returncode}", file=sys.stderr)
        return 1

    except FileNotFoundError:
        error_msg = "Zappa not found. Please install it: uv add zappa"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"Error during deployment: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_list(args: argparse.Namespace) -> int:  # noqa: PLR0912, C901
    """
    Handle the list command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get effective cache directory (with project prefix if specified)
        cache_dir = get_effective_cache_dir(args)

        # Load configuration
        config = load_config(cache_dir)
        models = config.get("models", {})

        if not models:
            print("No models configured.")
            print("\nTo add a model, run: python -m merle.cli prepare --model MODEL_NAME --auth-token TOKEN")
            return 0

        # Flatten nested structure for display and count total deployments
        model_stage_list = []
        for model_name, stages in models.items():
            for stage, stage_config in stages.items():
                model_stage_list.append((model_name, stage, stage_config))

        # Sort by model name then stage
        model_stage_list.sort(key=lambda x: (x[0], x[1]))

        print(f"Configured models in {cache_dir}:\n")
        print(f"{'Model':<20} {'Stage':<10} {'Status':<15} {'Region':<15} {'Auth Token':<20} {'System Prompt':<40}")
        print("=" * 130)

        for model_name, stage, model_config in model_stage_list:
            # Get model cache directory
            model_cache_dir = get_model_cache_dir(cache_dir, model_name, stage)

            # Check if deployment files exist
            zappa_settings = model_cache_dir / "zappa_settings.json"
            status = "Configured" if zappa_settings.exists() else "Not prepared"

            # Get region
            region = model_config.get("region", "N/A")

            # Get token (masked or raw)
            auth_token = model_config.get("auth_token", "N/A")  # noqa: S105
            if auth_token != "N/A" and not args.raw:  # noqa: S105
                auth_token = mask_token(auth_token)

            # Get system prompt (truncate if too long)
            system_prompt = model_config.get("system_prompt")
            if system_prompt:
                # Show first N-3 chars + "..." if longer than max display length
                max_len = MAX_SYSTEM_PROMPT_DISPLAY_LENGTH
                if len(system_prompt) > max_len:
                    system_prompt_display = system_prompt[: max_len - 3] + "..."
                else:
                    system_prompt_display = system_prompt
                # Replace newlines with space for display
                system_prompt_display = system_prompt_display.replace("\n", " ")
            else:
                system_prompt_display = "Not configured"

            # Get deployment URL - first from config, optionally verify with --check-urls
            url = model_config.get("url", "Not deployed")
            if status == "Configured":
                if url != "Not deployed":
                    status = "Deployed"
                if args.check_urls:
                    # Verify deployment status by querying AWS
                    logger.info(f"Checking deployment status for {model_name} (stage: {stage})...")
                    deployment_url = get_deployment_url(model_cache_dir)
                    if deployment_url:
                        url = deployment_url
                        status = "Deployed"
                    else:
                        url = "Not deployed"
                        status = "Configured"

            # Truncate model name if too long
            display_name = (
                model_name[: MAX_MODEL_NAME_LENGTH - 2] + ".."
                if len(model_name) > MAX_MODEL_NAME_LENGTH
                else model_name
            )

            print(
                f"{display_name:<20} {stage:<10} {status:<15} {region:<15} {auth_token:<20} {system_prompt_display:<40}"
            )

            # If there's a URL, print it on the next line with indentation
            if url and url != "Not deployed":
                print(f"{'':>20} URL: {url}")

        print(f"\n{len(model_stage_list)} deployment(s) configured")

        if not args.check_urls:
            print("\nNote: Use --check-urls to verify deployment status and get URLs (slower)")
        if not args.raw:
            print("Note: Use --raw to show unmasked authentication tokens")

        return 0

    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_destroy(args: argparse.Namespace) -> int:
    """
    Handle the destroy command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        logger.info("Tearing down deployment using Zappa")

        # Get effective cache directory (with project prefix if specified)
        cache_dir = get_effective_cache_dir(args)

        # Determine which model to use
        try:
            model_name = get_model_to_use(cache_dir, args.model)
        except ValueError as e:
            logger.error(str(e))
            print(f"Error: {e}", file=sys.stderr)
            return 1

        stage = args.stage
        logger.info(f"Destroying deployment for model: {model_name}, stage: {stage}")

        # Get model-stage-specific cache directory
        model_cache_dir = get_model_cache_dir(cache_dir, model_name, stage)
        zappa_settings = model_cache_dir / "zappa_settings.json"

        if not zappa_settings.exists():
            error_msg = f"No deployment found for model: {model_name}, stage: {stage}"
            logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return 1

        logger.info(f"Using zappa_settings.json from: {zappa_settings}")
        logger.info(f"Working directory: {model_cache_dir}")

        # Ask for confirmation
        if not args.yes:
            confirmation = input("Are you sure you want to tear down the deployment? (yes/no): ")
            if confirmation.lower() not in ["yes", "y"]:
                print("Aborted.")
                return 0

        # Run zappa undeploy (pass --yes to skip Zappa's confirmation)
        # User has already confirmed via --yes flag or by answering the prompt
        cmd = ["zappa", "undeploy", "dev", "--yes"]

        logger.info(f"Running: {' '.join(cmd)}")
        print(f"Tearing down deployment: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=model_cache_dir,
            check=False,
            capture_output=False,
        )

        # Check if undeploy succeeded or if there were no AWS resources to remove
        if result.returncode == 0:
            print("\nDeployment successfully torn down!")
        else:
            # Zappa undeploy failed - likely because resources don't exist
            logger.warning(f"Zappa undeploy failed with exit code: {result.returncode}")
            print(
                f"\nWarning: Zappa undeploy returned exit code {result.returncode}. "
                "This may indicate that AWS resources were already removed or never existed."
            )
            print("Proceeding to clean up local files...")

        # Clean up local files regardless of zappa undeploy result
        logger.info(f"Cleaning up local files for model: {model_name}, stage: {stage}")

        # Remove model-stage entry from config
        config = load_config(cache_dir)
        if model_name in config.get("models", {}) and stage in config["models"][model_name]:
            del config["models"][model_name][stage]
            # If no more stages for this model, remove the model entirely
            if not config["models"][model_name]:
                del config["models"][model_name]
            save_config(cache_dir, config)
            logger.info(f"Removed {model_name} (stage: {stage}) from configuration")

        # Delete model cache directory
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
            logger.info(f"Deleted model cache directory: {model_cache_dir}")
            print(f"Cleaned up local files in: {model_cache_dir}")

        print(f"\nModel '{model_name}' (stage: {stage}) has been completely removed.")
        return 0

    except FileNotFoundError:
        error_msg = "Zappa not found. Please install it: uv add zappa"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"Error during tear down: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_chat(args: argparse.Namespace) -> int:
    """
    Handle the chat command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        logger.info("Starting chat session")

        # Get effective cache directory (with project prefix if specified)
        cache_dir = get_effective_cache_dir(args)

        # Determine which model to use
        try:
            model_name = get_model_to_use(cache_dir, args.model)
        except ValueError as e:
            logger.error(str(e))
            print(f"Error: {e}", file=sys.stderr)
            return 1

        stage = args.stage
        logger.info(f"Connecting to model: {model_name}, stage: {stage}")

        # Get model-stage-specific cache directory
        model_cache_dir = get_model_cache_dir(cache_dir, model_name, stage)
        zappa_settings = model_cache_dir / "zappa_settings.json"

        if not zappa_settings.exists():
            error_msg = (
                f"Model not prepared: {model_name}, stage: {stage}. "
                f"Run 'merle prepare --model {model_name} --stage {stage}' first."
            )
            logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return 1

        # Get deployment URL
        logger.info("Checking deployment status...")
        deployment_url = get_deployment_url(model_cache_dir)

        if not deployment_url:
            error_msg = (
                f"Model not deployed: {model_name}, stage: {stage}. "
                f"Run 'merle deploy --model {model_name} --stage {stage} --auth-token TOKEN' first."
            )
            logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return 1

        # Get auth token, system prompt, and context window size from config
        config = load_config(cache_dir)
        model_config = config.get("models", {}).get(model_name, {}).get(stage, {})
        auth_token = model_config.get("auth_token")
        system_prompt = model_config.get("system_prompt")
        context_window_size = model_config.get("context_window_size")

        if not auth_token:
            error_msg = (
                f"No auth token found for model: {model_name}, stage: {stage}. "
                f"Please redeploy with --auth-token or update config.json."
            )
            logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return 1

        # Clean up deployment URL (remove trailing slash)
        deployment_url = deployment_url.rstrip("/")

        logger.info(f"Connected to: {deployment_url}")
        logger.info(f"Using model: {model_name}, stage: {stage}")
        if system_prompt:
            logger.info(f"Using system prompt ({len(system_prompt)} characters)")
        if context_window_size:
            logger.info(f"Context window size: {context_window_size} tokens")

        # Start interactive chat
        run_interactive_chat(
            base_url=deployment_url,
            auth_token=auth_token,
            model=model_name,
            debug=args.debug,
            system_prompt=system_prompt,
            context_window_size=context_window_size,
        )

        return 0

    except Exception as e:
        logger.exception(f"Error during chat: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="merle",
        description="CLI for deploying Ollama models to AWS Lambda using Zappa.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"merle {__version__}",
        help="Show the version of merle.",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )

    # prepare-dockerfile command
    prepare_parser = subparsers.add_parser(
        "prepare-dockerfile",
        aliases=["prepare"],
        help="Prepare deployment files (Dockerfile, zappa_settings.json, etc.)",
    )
    prepare_parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name (e.g., 'schroneko/gemma-2-2b-jpn-it')",
    )
    prepare_parser.add_argument(
        "--auth-token",
        help="Authentication token for API access (optional, can be set during deploy)",
    )
    prepare_parser.add_argument(
        "--system-prompt",
        help="System prompt for chat context (string or path to UTF-8 text file)",
    )
    prepare_parser.add_argument(
        "--region",
        default=REGION,
        help=f"AWS region (default: {REGION})",
    )
    prepare_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Base cache directory (default: ~/.merle, files stored in model subdirectory)",
    )
    prepare_parser.add_argument(
        "--tags",
        help="AWS resource tags in format: key1=value1,key2=value2 (optional)",
    )
    prepare_parser.add_argument(
        "--s3-bucket",
        help="S3 bucket name for Zappa deployment (default: zappa-merle-<short-uuid>)",
    )
    prepare_parser.add_argument(
        "--stage",
        default=STAGE,
        help=f"Deployment stage (default: {STAGE})",
    )
    prepare_parser.add_argument(
        "--memory-size",
        type=int,
        default=LAMBDA_MEMORY_SIZE_DEFAULT,
        help=f"Lambda function memory size in MB (default: {LAMBDA_MEMORY_SIZE_DEFAULT}, "
        f"min: {LAMBDA_MEMORY_SIZE_MIN}, max: {LAMBDA_MEMORY_SIZE_MAX})",
    )
    prepare_parser.add_argument(
        "--project",
        help="Project name to prefix cache directory (allows multiple projects to share the same base cache)",
    )
    prepare_parser.set_defaults(func=handle_prepare_dockerfile)

    # deploy command
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy the Ollama model server to AWS Lambda using Zappa",
    )
    deploy_parser.add_argument(
        "--auth-token",
        help="Authentication token for API access (uses existing token if already configured)",
    )
    deploy_parser.add_argument(
        "--system-prompt",
        help="System prompt for chat context (string or path to UTF-8 text file)",
    )
    deploy_parser.add_argument(
        "--model",
        help="Ollama model name to deploy (auto-detected if only one model exists)",
    )
    deploy_parser.add_argument(
        "--region",
        default=REGION,
        help=f"AWS region (default: {REGION}, only used if auto-preparing)",
    )
    deploy_parser.add_argument(
        "--cache-dir",
        help="Base cache directory (default: ~/.merle)",
    )
    deploy_parser.add_argument(
        "--tags",
        help="AWS resource tags in format: key1=value1,key2=value2 (optional, only used if auto-preparing)",
    )
    deploy_parser.add_argument(
        "--s3-bucket",
        help="S3 bucket name for Zappa deployment (default: zappa-merle-<short-uuid>)",
    )
    deploy_parser.add_argument(
        "--stage",
        default=STAGE,
        help=f"Deployment stage (default: {STAGE})",
    )
    deploy_parser.add_argument(
        "--memory-size",
        type=int,
        default=LAMBDA_MEMORY_SIZE_DEFAULT,
        help=f"Lambda function memory size in MB (default: {LAMBDA_MEMORY_SIZE_DEFAULT}, "
        f"min: {LAMBDA_MEMORY_SIZE_MIN}, max: {LAMBDA_MEMORY_SIZE_MAX}, only used if auto-preparing)",
    )
    deploy_parser.add_argument(
        "--project",
        help="Project name to prefix cache directory (allows multiple projects to share the same base cache)",
    )
    deploy_parser.set_defaults(func=handle_deploy)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List all configured models",
    )
    list_parser.add_argument(
        "--raw",
        action="store_true",
        help="Show unmasked authentication tokens",
    )
    list_parser.add_argument(
        "--check-urls",
        action="store_true",
        help="Check deployment status and get URLs (slower, requires zappa)",
    )
    list_parser.add_argument(
        "--cache-dir",
        help="Base cache directory (default: ~/.merle)",
    )
    list_parser.add_argument(
        "--project",
        help="Project name to prefix cache directory (allows multiple projects to share the same base cache)",
    )
    list_parser.set_defaults(func=handle_list)

    # destroy command
    destroy_parser = subparsers.add_parser(
        "destroy",
        aliases=["undeploy"],
        help="Tear down the deployed Lambda function",
    )
    destroy_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    destroy_parser.add_argument(
        "--model",
        help="Ollama model name to destroy (auto-detected if only one model exists)",
    )
    destroy_parser.add_argument(
        "--stage",
        default=STAGE,
        help=f"Deployment stage (default: {STAGE})",
    )
    destroy_parser.add_argument(
        "--cache-dir",
        help="Base cache directory (default: ~/.merle)",
    )
    destroy_parser.add_argument(
        "--project",
        help="Project name to prefix cache directory (allows multiple projects to share the same base cache)",
    )
    destroy_parser.set_defaults(func=handle_destroy)

    # chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session with a deployed model",
    )
    chat_parser.add_argument(
        "--model",
        help="Ollama model name to chat with (auto-detected if only one model exists)",
    )
    chat_parser.add_argument(
        "--stage",
        default=STAGE,
        help=f"Deployment stage (default: {STAGE})",
    )
    chat_parser.add_argument(
        "--cache-dir",
        help="Base cache directory (default: ~/.merle)",
    )
    chat_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug and info log messages during chat",
    )
    chat_parser.add_argument(
        "--project",
        help="Project name to prefix cache directory (allows multiple projects to share the same base cache)",
    )
    chat_parser.set_defaults(func=handle_chat)

    # Parse arguments and call appropriate handler
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
