"""Tests for merle.model_split module."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from merle.model_split import (
    MAX_MODEL_SIZE_IN_IMAGE,
    _get_model_blob_digests,
    _get_model_manifest_path,
    _get_platform_models_dirs,
    calculate_directory_size,
    calculate_model_size,
    calculate_split_sizes,
    find_largest_blob,
    get_ollama_models_dir,
    needs_splitting,
    reassemble_blob,
    split_blob_file,
)


class TestGetPlatformModelsDirs:
    """Tests for _get_platform_models_dirs."""

    @patch("merle.model_split.sys")
    def test_macos_returns_home_dir(self, mock_sys):
        mock_sys.platform = "darwin"
        dirs = _get_platform_models_dirs()
        assert len(dirs) == 1
        assert dirs[0] == Path.home() / ".ollama" / "models"

    @patch("merle.model_split.sys")
    def test_windows_returns_home_dir(self, mock_sys):
        mock_sys.platform = "win32"
        dirs = _get_platform_models_dirs()
        assert len(dirs) == 1
        assert dirs[0] == Path.home() / ".ollama" / "models"

    @patch("merle.model_split.sys")
    def test_linux_returns_systemd_and_user_dirs(self, mock_sys):
        mock_sys.platform = "linux"
        dirs = _get_platform_models_dirs()
        assert len(dirs) == 2
        assert dirs[0] == Path("/usr/share/ollama/.ollama/models")
        assert dirs[1] == Path.home() / ".ollama" / "models"


class TestGetOllamaModelsDir:
    """Tests for get_ollama_models_dir."""

    def test_env_variable_takes_precedence(self, tmp_path):
        env_dir = str(tmp_path / "custom_models")
        with patch.dict("os.environ", {"OLLAMA_MODELS": env_dir}):
            result = get_ollama_models_dir()
            assert result == Path(env_dir)

    def test_returns_existing_platform_dir(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_MODELS"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("merle.model_split._get_platform_models_dirs", return_value=[tmp_path]),
        ):
            result = get_ollama_models_dir()
            assert result == tmp_path

    def test_falls_back_to_first_default(self):
        non_existent = Path("/nonexistent/path/models")
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_MODELS"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("merle.model_split._get_platform_models_dirs", return_value=[non_existent]),
        ):
            result = get_ollama_models_dir()
            assert result == non_existent


class TestCalculateDirectorySize:
    """Tests for calculate_directory_size."""

    def test_empty_directory(self, tmp_path):
        assert calculate_directory_size(tmp_path) == 0

    def test_single_file(self, tmp_path):
        f = tmp_path / "file.bin"
        f.write_bytes(b"x" * 1024)
        assert calculate_directory_size(tmp_path) == 1024

    def test_nested_files(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        (sub / "b.bin").write_bytes(b"y" * 200)
        assert calculate_directory_size(tmp_path) == 300


class TestGetModelManifestPath:
    """Tests for _get_model_manifest_path."""

    def test_simple_model_name(self, tmp_path):
        manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "llama2" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        result = _get_model_manifest_path("llama2", tmp_path)
        assert result == manifest

    def test_model_with_tag(self, tmp_path):
        manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "llama2" / "7b"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        result = _get_model_manifest_path("llama2:7b", tmp_path)
        assert result == manifest

    def test_user_model(self, tmp_path):
        manifest = tmp_path / "manifests" / "registry.ollama.ai" / "myuser" / "mymodel" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        result = _get_model_manifest_path("myuser/mymodel", tmp_path)
        assert result == manifest

    def test_huggingface_model(self, tmp_path):
        manifest = tmp_path / "manifests" / "hf.co" / "org" / "model" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        result = _get_model_manifest_path("hf.co/org/model", tmp_path)
        assert result == manifest

    def test_huggingface_model_with_tag(self, tmp_path):
        manifest = tmp_path / "manifests" / "hf.co" / "org" / "model" / "Q4_K_M"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        result = _get_model_manifest_path("hf.co/org/model:Q4_K_M", tmp_path)
        assert result == manifest

    def test_nonexistent_model_returns_none(self, tmp_path):
        result = _get_model_manifest_path("nonexistent", tmp_path)
        assert result is None


class TestGetModelBlobDigests:
    """Tests for _get_model_blob_digests."""

    def test_extracts_config_and_layer_digests(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "config": {"digest": "sha256:config123"},
                    "layers": [
                        {"digest": "sha256:layer1abc"},
                        {"digest": "sha256:layer2def"},
                    ],
                }
            )
        )
        result = _get_model_blob_digests(manifest)
        assert result == ["sha256-config123", "sha256-layer1abc", "sha256-layer2def"]

    def test_handles_missing_config(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"layers": [{"digest": "sha256:abc"}]}))
        result = _get_model_blob_digests(manifest)
        assert result == ["sha256-abc"]

    def test_handles_empty_layers(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"config": {"digest": "sha256:cfg"}, "layers": []}))
        result = _get_model_blob_digests(manifest)
        assert result == ["sha256-cfg"]


class TestFindLargestBlob:
    """Tests for find_largest_blob."""

    def test_no_blobs_dir(self, tmp_path):
        assert find_largest_blob(tmp_path) is None

    def test_empty_blobs_dir(self, tmp_path):
        (tmp_path / "blobs").mkdir()
        assert find_largest_blob(tmp_path) is None

    def test_finds_largest(self, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "small").write_bytes(b"x" * 100)
        (blobs / "large").write_bytes(b"x" * 1000)
        (blobs / "medium").write_bytes(b"x" * 500)
        result = find_largest_blob(tmp_path)
        assert result is not None
        assert result[0].name == "large"
        assert result[1] == 1000

    def test_filter_by_model_name(self, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "sha256-abc").write_bytes(b"x" * 500)
        (blobs / "sha256-def").write_bytes(b"x" * 1000)

        # Create manifest referencing only sha256-abc
        manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "test" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"layers": [{"digest": "sha256:abc"}]}))

        result = find_largest_blob(tmp_path, model_name="test")
        assert result is not None
        assert result[0].name == "sha256-abc"
        assert result[1] == 500


class TestCalculateModelSize:
    """Tests for calculate_model_size."""

    def test_with_manifest(self, tmp_path):
        # Set up blobs
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "sha256-cfg").write_bytes(b"c" * 100)
        (blobs / "sha256-layer1").write_bytes(b"a" * 500)

        # Set up manifest
        manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "mymodel" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "config": {"digest": "sha256:cfg"},
                    "layers": [{"digest": "sha256:layer1"}],
                }
            )
        )

        with patch("merle.model_split.get_ollama_models_dir", return_value=tmp_path):
            total_size, details = calculate_model_size("mymodel")

        assert total_size == 600
        assert details["blob_count"] == 2
        assert details["largest_blob"]["name"] == "sha256-layer1"

    def test_fallback_without_manifest(self, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "sha256-abc").write_bytes(b"x" * 1024)

        with patch("merle.model_split.get_ollama_models_dir", return_value=tmp_path):
            total_size, details = calculate_model_size("nonexistent")

        assert total_size == 1024
        assert "manifest_path" not in details


class TestNeedsSplitting:
    """Tests for needs_splitting."""

    def test_small_model_no_split(self):
        assert needs_splitting(1024) is False

    def test_exact_limit_no_split(self):
        assert needs_splitting(MAX_MODEL_SIZE_IN_IMAGE) is False

    def test_over_limit_needs_split(self):
        assert needs_splitting(MAX_MODEL_SIZE_IN_IMAGE + 1) is True


class TestCalculateSplitSizes:
    """Tests for calculate_split_sizes."""

    def test_small_model_no_s3(self):
        image, s3 = calculate_split_sizes(1024)
        assert image == 1024
        assert s3 == 0

    def test_large_model_splits(self):
        total = MAX_MODEL_SIZE_IN_IMAGE + 2 * 1024 * 1024 * 1024  # 5GB + 2GB
        image, s3 = calculate_split_sizes(total)
        assert image == MAX_MODEL_SIZE_IN_IMAGE
        assert s3 == total - MAX_MODEL_SIZE_IN_IMAGE
        assert image + s3 == total


class TestSplitBlobFile:
    """Tests for split_blob_file."""

    def test_split_produces_two_parts(self, tmp_path):
        blob = tmp_path / "source" / "sha256-testblob"
        blob.parent.mkdir(parents=True)
        data = b"A" * 1000 + b"B" * 500
        blob.write_bytes(data)

        output_dir = tmp_path / "output"
        metadata = split_blob_file(blob, 1000, output_dir)

        assert metadata["original_blob_size"] == 1500
        assert metadata["part1"]["size"] == 1000
        assert metadata["part2"]["size"] == 500

        # Verify files exist
        part1 = output_dir / "blobs" / "sha256-testblob.part1"
        part2 = output_dir / "s3_overflow" / "sha256-testblob.part2"
        assert part1.exists()
        assert part2.exists()
        assert part1.read_bytes() == b"A" * 1000
        assert part2.read_bytes() == b"B" * 500

    def test_split_sha256_integrity(self, tmp_path):
        blob = tmp_path / "source" / "sha256-intblob"
        blob.parent.mkdir(parents=True)
        data = b"X" * 2048
        blob.write_bytes(data)

        expected_hash = hashlib.sha256(data).hexdigest()
        output_dir = tmp_path / "output"
        metadata = split_blob_file(blob, 1024, output_dir)

        assert metadata["original_blob_sha256"] == expected_hash

        # Verify part hashes
        part1_path = output_dir / "blobs" / "sha256-intblob.part1"
        part2_path = output_dir / "s3_overflow" / "sha256-intblob.part2"
        assert hashlib.sha256(part1_path.read_bytes()).hexdigest() == metadata["part1"]["sha256"]
        assert hashlib.sha256(part2_path.read_bytes()).hexdigest() == metadata["part2"]["sha256"]


class TestReassembleBlob:
    """Tests for reassemble_blob."""

    def test_reassemble_matches_original(self, tmp_path):
        original_data = b"Hello" * 200
        part1 = tmp_path / "part1"
        part2 = tmp_path / "part2"
        part1.write_bytes(original_data[:500])
        part2.write_bytes(original_data[500:])

        output = tmp_path / "reassembled"
        expected_sha = hashlib.sha256(original_data).hexdigest()

        result = reassemble_blob(part1, part2, output, expected_sha256=expected_sha)
        assert result is True
        assert output.read_bytes() == original_data

    def test_reassemble_sha256_mismatch(self, tmp_path):
        part1 = tmp_path / "part1"
        part2 = tmp_path / "part2"
        part1.write_bytes(b"aaa")
        part2.write_bytes(b"bbb")

        output = tmp_path / "reassembled"
        result = reassemble_blob(part1, part2, output, expected_sha256="wrong_hash")
        assert result is False
        assert not output.exists()

    def test_reassemble_without_sha_verification(self, tmp_path):
        part1 = tmp_path / "part1"
        part2 = tmp_path / "part2"
        part1.write_bytes(b"abc")
        part2.write_bytes(b"def")

        output = tmp_path / "reassembled"
        result = reassemble_blob(part1, part2, output)
        assert result is True
        assert output.read_bytes() == b"abcdef"


class TestSplitAndReassembleRoundTrip:
    """Integration test: split a blob and reassemble it."""

    def test_roundtrip_integrity(self, tmp_path):
        # Create original blob
        blob = tmp_path / "source" / "sha256-roundtrip"
        blob.parent.mkdir(parents=True)
        original_data = bytes(range(256)) * 100  # 25.6KB of varied data
        blob.write_bytes(original_data)

        # Split
        output_dir = tmp_path / "split_output"
        metadata = split_blob_file(blob, len(original_data) // 3, output_dir)

        # Reassemble
        part1 = output_dir / "blobs" / metadata["part1"]["filename"]
        part2 = output_dir / "s3_overflow" / metadata["part2"]["filename"]
        reassembled = tmp_path / "reassembled"

        result = reassemble_blob(part1, part2, reassembled, expected_sha256=metadata["original_blob_sha256"])
        assert result is True
        assert reassembled.read_bytes() == original_data
