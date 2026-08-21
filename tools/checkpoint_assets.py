"""Load the shared checkpoint-assets implementation without importing ComfyUI nodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "comfy_easy_sensenova_u1"
    / "checkpoint_assets.py"
)
_SPEC = importlib.util.spec_from_file_location("_sensenova_checkpoint_assets", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load checkpoint assets module: {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ASSETS_ARCHIVE_KEY = _MODULE.ASSETS_ARCHIVE_KEY
ASSETS_FILE_COUNT_KEY = _MODULE.ASSETS_FILE_COUNT_KEY
ASSETS_FORMAT = _MODULE.ASSETS_FORMAT
ASSETS_FORMAT_KEY = _MODULE.ASSETS_FORMAT_KEY
ASSETS_SHA256_KEY = _MODULE.ASSETS_SHA256_KEY
build_assets_metadata = _MODULE.build_assets_metadata
materialize_checkpoint_assets = _MODULE.materialize_checkpoint_assets

__all__ = [
    "ASSETS_ARCHIVE_KEY",
    "ASSETS_FILE_COUNT_KEY",
    "ASSETS_FORMAT",
    "ASSETS_FORMAT_KEY",
    "ASSETS_SHA256_KEY",
    "build_assets_metadata",
    "materialize_checkpoint_assets",
]
