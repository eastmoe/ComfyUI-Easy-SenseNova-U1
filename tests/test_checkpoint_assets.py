from __future__ import annotations

import base64
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from safetensors import safe_open


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "tools"))

from checkpoint_assets import (  # noqa: E402
    ASSETS_ARCHIVE_KEY,
    ASSETS_FORMAT,
    ASSETS_FORMAT_KEY,
    build_assets_metadata,
    materialize_checkpoint_assets,
)


def write_minimal_bf16_checkpoint(path: Path) -> None:
    header = {
        "model.weight": {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [0, 2],
        }
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(raw_header)))
        output.write(raw_header)
        output.write(b"\x00\x00")


class EmbeddedCheckpointAssetsTest(unittest.TestCase):
    def test_converter_embeds_and_loader_materializes_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hf"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
            (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (source / "vocab.json").write_text('{"hello":0}', encoding="utf-8")
            write_minimal_bf16_checkpoint(source / "model.safetensors")
            checkpoint = root / "tiny.safetensors"

            subprocess.run(
                [
                    sys.executable,
                    str(PLUGIN_ROOT / "tools" / "convert_hf_to_comfy_checkpoint.py"),
                    str(source),
                    str(checkpoint),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(checkpoint.is_file())
            self.assertFalse((root / "tiny_assets").exists())
            with safe_open(checkpoint, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                self.assertEqual(metadata[ASSETS_FORMAT_KEY], ASSETS_FORMAT)
                self.assertTrue(metadata[ASSETS_ARCHIVE_KEY])
                self.assertIn("vae.pixel_space_vae", set(handle.keys()))

            cache_root = root / "ComfyUI" / "temp"
            assets = materialize_checkpoint_assets(checkpoint, metadata, cache_root)
            self.assertEqual(assets, cache_root / "tiny_assets")
            self.assertEqual((assets / "config.json").read_text(encoding="utf-8"), '{"model_type":"test"}')
            self.assertEqual((assets / "vocab.json").read_text(encoding="utf-8"), '{"hello":0}')
            self.assertTrue(os.path.samefile(assets / "model.safetensors", checkpoint))

            manifest_mtime = (assets / ".sensenova-assets.json").stat().st_mtime_ns
            self.assertEqual(materialize_checkpoint_assets(checkpoint, metadata, cache_root), assets)
            self.assertEqual((assets / ".sensenova-assets.json").stat().st_mtime_ns, manifest_mtime)

    def test_legacy_sidecar_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "legacy.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "Reconvert"):
                materialize_checkpoint_assets(
                    checkpoint,
                    {"sensenova_assets_dir": "legacy_assets"},
                    root / "temp",
                )

    def test_hardlink_fallback_does_not_copy_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assets"
            source.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            metadata = build_assets_metadata(source)

            with patch.object(Path, "symlink_to", side_effect=OSError("not permitted")):
                assets = materialize_checkpoint_assets(checkpoint, metadata, root / "temp")

            model_entry = assets / "model.safetensors"
            self.assertFalse(model_entry.is_symlink())
            self.assertTrue(os.path.samefile(model_entry, checkpoint))
            self.assertEqual(model_entry.stat().st_ino, checkpoint.stat().st_ino)

    def test_cache_manifest_is_not_reembedded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / ".sensenova-assets.json").write_text(
                '{"checkpoint":"private/local/path"}',
                encoding="utf-8",
            )
            metadata = build_assets_metadata(source)
            payload = base64.b64decode(metadata[ASSETS_ARCHIVE_KEY])
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                self.assertEqual(archive.namelist(), ["config.json"])


if __name__ == "__main__":
    unittest.main()
