"""Model splitting logic for hybrid Docker image + S3 deployment.

This module handles:
1. Downloading models via local Ollama
2. Calculating model sizes
3. Splitting large models into image + S3 portions
4. Uploading overflow portions to S3
5. Reassembling models at Lambda runtime
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# HTTP status codes
HTTP_OK = 200

# Docker image size limits
# Lambda container image max is 10GB uncompressed
# Base image (Python runtime, Ollama, dependencies) is ~5GB
# This leaves ~5GB for the model in the Docker image
DOCKER_IMAGE_MAX_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB Lambda limit
BASE_IMAGE_OVERHEAD_BYTES = 5 * 1024 * 1024 * 1024  # ~5GB for base image + deps
MAX_MODEL_SIZE_IN_IMAGE = DOCKER_IMAGE_MAX_SIZE_BYTES - BASE_IMAGE_OVERHEAD_BYTES  # ~5GB for model

# Split metadata filename
SPLIT_METADATA_FILE = "split_metadata.json"


def _get_platform_models_dirs() -> list[Path]:
    """
    Get platform-specific Ollama models directories to search.

    Returns directories in priority order:
    - macOS: ~/.ollama/models
    - Linux: /usr/share/ollama/.ollama/models (systemd), ~/.ollama/models (user)
    - Windows: ~/.ollama/models
    """
    if sys.platform == "darwin":
        # macOS: ~/.ollama/models
        return [Path.home() / ".ollama" / "models"]

    if sys.platform == "win32":
        # Windows: C:\Users\%username%\.ollama\models
        return [Path.home() / ".ollama" / "models"]

    # Linux: check systemd install first, then user install
    return [
        Path("/usr/share/ollama/.ollama/models"),  # systemd install
        Path.home() / ".ollama" / "models",  # user install
    ]


def get_ollama_models_dir() -> Path:
    """
    Get the Ollama models directory.

    Checks in order:
    1. OLLAMA_MODELS environment variable (if set)
    2. Platform-specific directories (first existing one):
       - macOS: ~/.ollama/models
       - Linux: /usr/share/ollama/.ollama/models (systemd) or ~/.ollama/models (user)
       - Windows: ~/.ollama/models

    Returns:
        Path to the Ollama models directory
    """
    # Check environment variable first
    env_models_dir = os.environ.get("OLLAMA_MODELS")
    if env_models_dir:
        models_path = Path(env_models_dir)
        logger.debug(f"Using OLLAMA_MODELS from environment: {models_path}")
        return models_path

    # Search platform-specific directories
    for models_dir in _get_platform_models_dirs():
        if models_dir.exists():
            logger.debug(f"Found models directory: {models_dir}")
            return models_dir

    # Fall back to first platform default (even if it doesn't exist yet)
    default_dir = _get_platform_models_dirs()[0]
    logger.debug(f"Using default models directory: {default_dir}")
    return default_dir


def calculate_directory_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def find_largest_blob(models_dir: Path, model_name: str | None = None) -> tuple[Path, int] | None:
    """
    Find the largest blob file for a model.

    Ollama stores model weights in the blobs directory as SHA256-named files.

    Args:
        models_dir: Path to the Ollama models directory
        model_name: Optional model name to filter blobs (if None, searches all blobs)

    Returns:
        Tuple of (blob_path, size) or None if no blobs found
    """
    blobs_dir = models_dir / "blobs"
    if not blobs_dir.exists():
        return None

    # If model name provided, get only that model's blobs
    if model_name:
        manifest_path = _get_model_manifest_path(model_name, models_dir)
        if manifest_path:
            digests = _get_model_blob_digests(manifest_path)
            blob_files = [blobs_dir / digest for digest in digests if (blobs_dir / digest).exists()]
        else:
            # Fallback to all blobs if manifest not found
            blob_files = [f for f in blobs_dir.iterdir() if f.is_file()]
    else:
        blob_files = [f for f in blobs_dir.iterdir() if f.is_file()]

    largest_blob = None
    largest_size = 0

    for blob_file in blob_files:
        if blob_file.is_file():
            size = blob_file.stat().st_size
            if size > largest_size:
                largest_size = size
                largest_blob = blob_file

    if largest_blob:
        return (largest_blob, largest_size)
    return None


def download_model_locally(model_name: str, timeout: int = 1800) -> Path:
    """
    Download a model using the local Ollama instance.

    Args:
        model_name: Name of the model to download
        timeout: Timeout in seconds (default 30 minutes)

    Returns:
        Path to the Ollama models directory

    Raises:
        RuntimeError: If Ollama is not available or download fails
    """
    import httpx  # noqa: PLC0415

    # Check if Ollama is running locally
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code != HTTP_OK:
            raise RuntimeError("Local Ollama server not responding correctly")
    except httpx.ConnectError as e:
        raise RuntimeError("Local Ollama server not running. Please start Ollama with 'ollama serve' first.") from e

    logger.info(f"Downloading model {model_name} using local Ollama...")

    # Pull the model - ollama is a trusted command, model_name is validated upstream
    result = subprocess.run(  # noqa: S603
        ["ollama", "pull", model_name],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull model: {result.stderr}")

    logger.info(f"Model {model_name} downloaded successfully")
    return get_ollama_models_dir()


def get_model_info(model_name: str) -> dict | None:
    """
    Get information about a model from local Ollama.

    Args:
        model_name: Name of the model

    Returns:
        Model info dict or None if not found
    """
    import httpx  # noqa: PLC0415

    try:
        response = httpx.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=30.0,
        )
        if response.status_code == HTTP_OK:
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
        logger.warning(f"Failed to get model info: {e}")
    return None


def _get_model_manifest_path(model_name: str, models_dir: Path) -> Path | None:
    """
    Get the manifest file path for a model.

    Handles different model name formats:
    - Simple: "llama2" → manifests/registry.ollama.ai/library/llama2/latest
    - With tag: "llama2:7b" → manifests/registry.ollama.ai/library/llama2/7b
    - User model: "user/model" → manifests/registry.ollama.ai/user/model/latest
    - HuggingFace: "hf.co/org/model" → manifests/hf.co/org/model/latest

    Args:
        model_name: Ollama model name
        models_dir: Path to Ollama models directory

    Returns:
        Path to manifest file or None if not found
    """
    # Parse model name into components
    # Format: [registry/][namespace/]model[:tag]
    parts = model_name.split(":")
    name_part = parts[0]
    tag = parts[1] if len(parts) > 1 else "latest"

    # Check if it's a HuggingFace model (hf.co/...)
    if name_part.startswith("hf.co/"):
        # HuggingFace models: hf.co/org/model → manifests/hf.co/org/model/tag
        manifest_path = models_dir / "manifests" / name_part / tag
        if manifest_path.exists():
            return manifest_path
        return None

    # Standard Ollama models use registry.ollama.ai
    if "/" in name_part:
        # User model: user/model → registry.ollama.ai/user/model/tag
        namespace, model = name_part.split("/", 1)
    else:
        # Library model: model → registry.ollama.ai/library/model/tag
        namespace = "library"
        model = name_part

    manifest_path = models_dir / "manifests" / "registry.ollama.ai" / namespace / model / tag
    if manifest_path.exists():
        return manifest_path

    return None


def _get_model_blob_digests(manifest_path: Path) -> list[str]:
    """
    Parse a manifest file to get the list of blob digests.

    Args:
        manifest_path: Path to the manifest file

    Returns:
        List of blob digest strings (e.g., ["sha256-abc123..."])
    """
    with manifest_path.open() as f:
        manifest = json.load(f)

    digests = []

    # Add config digest
    config = manifest.get("config", {})
    if "digest" in config:
        # Convert sha256:abc to sha256-abc (file naming convention)
        digests.append(config["digest"].replace(":", "-"))

    # Add layer digests
    for layer in manifest.get("layers", []):
        if "digest" in layer:
            digests.append(layer["digest"].replace(":", "-"))

    return digests


def calculate_model_size(model_name: str) -> tuple[int, dict]:
    """
    Calculate the size of a specific model and return size details.

    Parses the model's manifest to find its blob references and sums
    only those blobs, not the entire models directory.

    Args:
        model_name: Name of the model to calculate size for

    Returns:
        Tuple of (total_size_bytes, size_details_dict)
    """
    models_dir = get_ollama_models_dir()

    # Try to find model manifest
    manifest_path = _get_model_manifest_path(model_name, models_dir)

    if manifest_path:
        # Calculate size from manifest blob references
        digests = _get_model_blob_digests(manifest_path)
        blobs_dir = models_dir / "blobs"

        total_size = 0
        largest_blob_path = None
        largest_blob_size = 0

        for digest in digests:
            blob_path = blobs_dir / digest
            if blob_path.exists():
                blob_size = blob_path.stat().st_size
                total_size += blob_size
                if blob_size > largest_blob_size:
                    largest_blob_size = blob_size
                    largest_blob_path = blob_path

        size_details = {
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
            "models_dir": str(models_dir),
            "manifest_path": str(manifest_path),
            "blob_count": len(digests),
        }

        if largest_blob_path:
            size_details["largest_blob"] = {
                "path": str(largest_blob_path),
                "name": largest_blob_path.name,
                "size_bytes": largest_blob_size,
                "size_gb": round(largest_blob_size / (1024 * 1024 * 1024), 2),
            }

        return total_size, size_details

    # Fallback: calculate entire directory size (legacy behavior)
    logger.warning(f"Could not find manifest for {model_name}, calculating total directory size")
    total_size = calculate_directory_size(models_dir)

    # Find largest blob
    blob_info = find_largest_blob(models_dir)

    size_details = {
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
        "models_dir": str(models_dir),
    }

    if blob_info:
        blob_path, blob_size = blob_info
        size_details["largest_blob"] = {
            "path": str(blob_path),
            "name": blob_path.name,
            "size_bytes": blob_size,
            "size_gb": round(blob_size / (1024 * 1024 * 1024), 2),
        }

    return total_size, size_details


def needs_splitting(total_size: int) -> bool:
    """Check if a model needs to be split based on size."""
    return total_size > MAX_MODEL_SIZE_IN_IMAGE


def calculate_split_sizes(total_size: int) -> tuple[int, int]:
    """
    Calculate how to split a model between image and S3.

    The goal is to maximize what fits in the Docker image to minimize
    cold start download time.

    Args:
        total_size: Total model size in bytes

    Returns:
        Tuple of (image_portion_bytes, s3_portion_bytes)
    """
    if total_size <= MAX_MODEL_SIZE_IN_IMAGE:
        return (total_size, 0)

    # Fill image to near max, put rest in S3
    image_portion = MAX_MODEL_SIZE_IN_IMAGE
    s3_portion = total_size - image_portion

    return (image_portion, s3_portion)


def split_blob_file(
    blob_path: Path,
    image_portion_bytes: int,
    output_dir: Path,
) -> dict:
    """
    Split a blob file into two parts.

    Part 1 goes into the Docker image.
    Part 2 will be uploaded to S3.

    Args:
        blob_path: Path to the blob file to split
        image_portion_bytes: How many bytes to keep in image
        output_dir: Directory to write split files

    Returns:
        Split metadata dict
    """
    blob_size = blob_path.stat().st_size
    blob_name = blob_path.name

    # Ensure output directories exist
    part1_dir = output_dir / "blobs"
    part2_dir = output_dir / "s3_overflow"
    part1_dir.mkdir(parents=True, exist_ok=True)
    part2_dir.mkdir(parents=True, exist_ok=True)

    part1_path = part1_dir / f"{blob_name}.part1"
    part2_path = part2_dir / f"{blob_name}.part2"

    logger.info(f"Splitting blob {blob_name} ({blob_size} bytes)")
    logger.info(f"  Part 1 (image): {image_portion_bytes} bytes")
    logger.info(f"  Part 2 (S3): {blob_size - image_portion_bytes} bytes")

    # Calculate checksums while splitting
    part1_hash = hashlib.sha256()
    part2_hash = hashlib.sha256()
    full_hash = hashlib.sha256()

    chunk_size = 64 * 1024 * 1024  # 64MB chunks for efficient I/O
    bytes_written_part1 = 0

    with blob_path.open("rb") as src, part1_path.open("wb") as p1, part2_path.open("wb") as p2:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break

            full_hash.update(chunk)

            if bytes_written_part1 < image_portion_bytes:
                # Still writing to part 1
                remaining = image_portion_bytes - bytes_written_part1
                if len(chunk) <= remaining:
                    p1.write(chunk)
                    part1_hash.update(chunk)
                    bytes_written_part1 += len(chunk)
                else:
                    # Split this chunk
                    p1.write(chunk[:remaining])
                    part1_hash.update(chunk[:remaining])
                    bytes_written_part1 += remaining

                    p2.write(chunk[remaining:])
                    part2_hash.update(chunk[remaining:])
            else:
                # Writing to part 2
                p2.write(chunk)
                part2_hash.update(chunk)

    metadata = {
        "original_blob_name": blob_name,
        "original_blob_size": blob_size,
        "original_blob_sha256": full_hash.hexdigest(),
        "part1": {
            "filename": f"{blob_name}.part1",
            "size": part1_path.stat().st_size,
            "sha256": part1_hash.hexdigest(),
        },
        "part2": {
            "filename": f"{blob_name}.part2",
            "size": part2_path.stat().st_size,
            "sha256": part2_hash.hexdigest(),
        },
    }

    logger.info(f"Split complete. Part 1: {metadata['part1']['size']}, Part 2: {metadata['part2']['size']}")
    return metadata


def upload_to_s3(
    file_path: Path,
    bucket: str,
    key: str,
    region: str,
) -> str:
    """
    Upload a file to S3.

    Args:
        file_path: Path to file to upload
        bucket: S3 bucket name
        key: S3 object key
        region: AWS region

    Returns:
        S3 URI (s3://bucket/key)
    """
    s3 = boto3.client("s3", region_name=region)

    # Create bucket if it doesn't exist
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.info(f"Creating S3 bucket: {bucket}")
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
        else:
            raise

    # Upload with progress
    file_size = file_path.stat().st_size
    logger.info(f"Uploading {file_path.name} to s3://{bucket}/{key} ({file_size} bytes)")

    s3.upload_file(str(file_path), bucket, key)

    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Upload complete: {s3_uri}")
    return s3_uri


def download_from_s3(
    bucket: str,
    key: str,
    dest_path: Path,
    region: str,
) -> None:
    """
    Download a file from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        dest_path: Local destination path
        region: AWS region
    """
    s3 = boto3.client("s3", region_name=region)

    logger.info(f"Downloading s3://{bucket}/{key} to {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    s3.download_file(bucket, key, str(dest_path))
    logger.info(f"Download complete: {dest_path}")


def reassemble_blob(
    part1_path: Path,
    part2_path: Path,
    output_path: Path,
    expected_sha256: str | None = None,
) -> bool:
    """
    Reassemble a split blob file from its parts.

    Args:
        part1_path: Path to part 1 file
        part2_path: Path to part 2 file
        output_path: Path for reassembled file
        expected_sha256: Optional SHA256 hash to verify

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Reassembling blob from {part1_path} and {part2_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256() if expected_sha256 else None

    chunk_size = 64 * 1024 * 1024  # 64MB chunks

    with output_path.open("wb") as out:
        for part_path in [part1_path, part2_path]:
            with part_path.open("rb") as part:
                while True:
                    chunk = part.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    if hasher:
                        hasher.update(chunk)

    if expected_sha256 and hasher:
        actual_sha256 = hasher.hexdigest()
        if actual_sha256 != expected_sha256:
            logger.error(f"SHA256 mismatch! Expected {expected_sha256}, got {actual_sha256}")
            output_path.unlink()
            return False
        logger.info("SHA256 verification passed")

    logger.info(f"Reassembly complete: {output_path} ({output_path.stat().st_size} bytes)")
    return True


def copy_model_to_output(model_name: str, output_dir: Path) -> dict:
    """
    Copy model files to output directory for Docker image inclusion.

    Used when model fits in Docker image (no splitting needed).
    Only copies the blobs referenced by the specific model, not all blobs.

    Args:
        model_name: Ollama model name
        output_dir: Directory to store model files

    Returns:
        Metadata dict with model info
    """
    models_dir = get_ollama_models_dir()
    total_size, size_details = calculate_model_size(model_name)

    # Prepare output directories
    models_output = output_dir / "models"
    blobs_output = models_output / "blobs"
    blobs_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying model files to {models_output}")

    # Get model-specific blobs from manifest
    manifest_path = _get_model_manifest_path(model_name, models_dir)
    blobs_src = models_dir / "blobs"

    if manifest_path:
        # Copy only model-specific blobs
        digests = _get_model_blob_digests(manifest_path)
        for digest in digests:
            src_blob = blobs_src / digest
            if src_blob.exists():
                shutil.copy2(src_blob, blobs_output / digest)
        logger.info(f"Copied {len(digests)} model-specific blobs")

        # Copy model-specific manifest
        manifests_output = models_output / "manifests"
        manifest_rel = manifest_path.relative_to(models_dir / "manifests")
        dest_manifest = manifests_output / manifest_rel
        dest_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, dest_manifest)
        logger.info(f"Copied manifest: {manifest_rel}")
    else:
        # Fallback: copy all blobs and manifests (legacy behavior)
        logger.warning(f"Could not find manifest for {model_name}, copying all files")
        if blobs_src.exists():
            shutil.copytree(blobs_src, blobs_output, dirs_exist_ok=True)
        manifests_src = models_dir / "manifests"
        if manifests_src.exists():
            shutil.copytree(manifests_src, models_output / "manifests", dirs_exist_ok=True)

    return {
        "split_required": False,
        "model_name": model_name,
        "total_size_bytes": total_size,
        "total_size_gb": size_details["total_size_gb"],
        "models_dir": str(models_output),
    }


def prepare_split_model(  # noqa: PLR0915
    model_name: str,
    output_dir: Path,
    s3_bucket: str,
    region: str,
) -> dict:
    """
    Prepare a model for split deployment.

    This is the main entry point for the prepare command.
    Only copies blobs referenced by the specific model, not all blobs.

    Args:
        model_name: Ollama model name
        output_dir: Directory to store prepared files
        s3_bucket: S3 bucket for overflow storage
        region: AWS region

    Returns:
        Split metadata dict with all necessary info for deployment
    """
    # Download the model
    models_dir = download_model_locally(model_name)

    # Calculate size (model-specific)
    total_size, size_details = calculate_model_size(model_name)
    logger.info(f"Model size: {size_details['total_size_gb']} GB")

    # Prepare output directories
    models_output = output_dir / "models"
    blobs_output = models_output / "blobs"
    blobs_output.mkdir(parents=True, exist_ok=True)

    # Get model-specific manifest and blobs
    manifest_path = _get_model_manifest_path(model_name, models_dir)
    blobs_src = models_dir / "blobs"

    if not manifest_path:
        raise RuntimeError(f"Could not find manifest for model {model_name}")

    model_digests = _get_model_blob_digests(manifest_path)

    if not needs_splitting(total_size):
        # Model fits in image - just copy model-specific files
        logger.info("Model fits in Docker image, no splitting needed")

        # Copy model-specific blobs
        for digest in model_digests:
            src_blob = blobs_src / digest
            if src_blob.exists():
                shutil.copy2(src_blob, blobs_output / digest)
        logger.info(f"Copied {len(model_digests)} model-specific blobs")

        # Copy model-specific manifest
        manifests_output = models_output / "manifests"
        manifest_rel = manifest_path.relative_to(models_dir / "manifests")
        dest_manifest = manifests_output / manifest_rel
        dest_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, dest_manifest)
        logger.info(f"Copied manifest: {manifest_rel}")

        return {
            "split_required": False,
            "model_name": model_name,
            "total_size_bytes": total_size,
            "total_size_gb": size_details["total_size_gb"],
            "models_dir": str(models_output),
        }

    # Model needs splitting
    logger.info("Model exceeds Docker image limit, splitting required")

    blob_info = find_largest_blob(models_dir, model_name)
    if not blob_info:
        raise RuntimeError("Could not find blob files for model")

    blob_path, blob_size = blob_info
    image_portion, s3_portion = calculate_split_sizes(total_size)

    # For the blob, calculate how much of it fits in image
    # (total model size - other files) determines blob portion
    other_files_size = total_size - blob_size
    blob_image_portion = image_portion - other_files_size

    if blob_image_portion <= 0:
        raise RuntimeError(
            f"Other model files ({other_files_size} bytes) exceed image limit. "
            "This model is too large for Lambda deployment."
        )

    # Split the blob
    split_metadata = split_blob_file(blob_path, blob_image_portion, output_dir)

    # Copy model-specific manifest
    manifests_output = models_output / "manifests"
    manifest_rel = manifest_path.relative_to(models_dir / "manifests")
    dest_manifest = manifests_output / manifest_rel
    dest_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, dest_manifest)
    logger.info(f"Copied manifest: {manifest_rel}")

    # Copy other model-specific blobs (not the one we split)
    for digest in model_digests:
        src_blob = blobs_src / digest
        if src_blob.exists() and src_blob.name != blob_path.name:
            shutil.copy2(src_blob, blobs_output / digest)

    # Copy part 1 to blobs directory with .part1 extension
    part1_src = output_dir / "blobs" / split_metadata["part1"]["filename"]
    shutil.copy2(part1_src, blobs_output / split_metadata["part1"]["filename"])

    # Upload part 2 to S3
    part2_src = output_dir / "s3_overflow" / split_metadata["part2"]["filename"]
    s3_key = f"merle-models/{model_name.replace('/', '_')}/{split_metadata['part2']['filename']}"
    s3_uri = upload_to_s3(part2_src, s3_bucket, s3_key, region)

    # Create full metadata
    full_metadata = {
        "split_required": True,
        "model_name": model_name,
        "total_size_bytes": total_size,
        "total_size_gb": size_details["total_size_gb"],
        "image_portion_bytes": image_portion,
        "s3_portion_bytes": s3_portion,
        "models_dir": str(models_output),
        "blob_split": split_metadata,
        "s3": {
            "bucket": s3_bucket,
            "key": s3_key,
            "uri": s3_uri,
            "region": region,
        },
    }

    # Write metadata file for runtime use
    metadata_path = models_output / SPLIT_METADATA_FILE
    with metadata_path.open("w") as f:
        json.dump(full_metadata, f, indent=2)

    logger.info(f"Split model prepared. Metadata: {metadata_path}")
    return full_metadata


def reassemble_at_runtime(
    source_models_dir: Path,
    target_models_dir: Path,
    region: str | None = None,
) -> bool:
    """
    Reassemble a split model at Lambda runtime.

    This function should be called during Lambda cold start if split_metadata.json exists.

    Args:
        source_models_dir: Path to models in Docker image (/var/task/models)
        target_models_dir: Path to writable models dir (/tmp/models)
        region: AWS region (optional, will use env var if not provided)

    Returns:
        True if successful, False otherwise
    """
    metadata_path = source_models_dir / SPLIT_METADATA_FILE

    if not metadata_path.exists():
        # Not a split model, use normal initialization
        return True

    with metadata_path.open() as f:
        metadata = json.load(f)

    if not metadata.get("split_required"):
        return True

    logger.info("Split model detected, reassembling...")

    # Create target directory
    target_models_dir.mkdir(parents=True, exist_ok=True)

    # Copy manifests
    manifests_src = source_models_dir / "manifests"
    manifests_dst = target_models_dir / "manifests"
    if manifests_src.exists() and not manifests_dst.exists():
        shutil.copytree(manifests_src, manifests_dst)

    # Copy non-split blobs
    blobs_src = source_models_dir / "blobs"
    blobs_dst = target_models_dir / "blobs"
    blobs_dst.mkdir(parents=True, exist_ok=True)

    split_blob_name = metadata["blob_split"]["original_blob_name"]
    part1_filename = metadata["blob_split"]["part1"]["filename"]

    for blob_file in blobs_src.iterdir():
        if blob_file.is_file():
            if blob_file.name == part1_filename:
                # This is part 1 of the split blob, skip for now
                continue
            # Copy other blobs
            dest = blobs_dst / blob_file.name
            if not dest.exists():
                shutil.copy2(blob_file, dest)

    # Download part 2 from S3
    s3_info = metadata["s3"]
    effective_region = region or s3_info.get("region") or os.environ.get("AWS_REGION") or "us-east-1"

    part2_temp = target_models_dir / "temp_part2"
    download_from_s3(s3_info["bucket"], s3_info["key"], part2_temp, effective_region)

    # Reassemble the blob
    part1_path = blobs_src / part1_filename
    output_path = blobs_dst / split_blob_name
    expected_sha256 = metadata["blob_split"]["original_blob_sha256"]

    success = reassemble_blob(part1_path, part2_temp, output_path, expected_sha256)

    # Clean up temp file
    if part2_temp.exists():
        part2_temp.unlink()

    if success:
        logger.info("Model reassembly complete")
    else:
        logger.error("Model reassembly failed")

    return success
