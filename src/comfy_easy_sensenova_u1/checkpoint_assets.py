"""Embed and materialize the non-weight files required by SenseNova checkpoints."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath


ASSETS_ARCHIVE_KEY = "sensenova_assets_archive"
ASSETS_FORMAT_KEY = "sensenova_assets_format"
ASSETS_FORMAT = "zip+base64-v1"
ASSETS_SHA256_KEY = "sensenova_assets_sha256"
ASSETS_FILE_COUNT_KEY = "sensenova_assets_file_count"
WEIGHT_SUFFIXES = {".safetensors", ".sft", ".bin", ".pt", ".pth", ".ckpt"}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
MAX_ASSET_FILES = 512
MANIFEST_NAME = ".sensenova-assets.json"

_CACHE_LOCK = threading.RLock()


def _asset_files(source: Path) -> list[Path]:
    files = []
    for path in sorted(source.iterdir(), key=lambda item: item.name):
        if path.name == MANIFEST_NAME:
            continue
        if not path.is_file() or path.suffix.lower() in WEIGHT_SUFFIXES:
            continue
        if path.name.endswith(".safetensors.index.json"):
            continue
        files.append(path)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in assets source: {source}")
    if len(files) > MAX_ASSET_FILES:
        raise ValueError(f"Too many checkpoint asset files: {len(files)} > {MAX_ASSET_FILES}")
    return files


def build_assets_metadata(source: Path) -> dict[str, str]:
    """Compress top-level Hugging Face config/tokenizer files for safetensors metadata."""
    source = source.expanduser().resolve()
    buffer = io.BytesIO()
    files = _asset_files(source)
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=False,
    ) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    payload = buffer.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Compressed checkpoint assets are too large: {len(payload)} > {MAX_ARCHIVE_BYTES} bytes"
        )
    return {
        ASSETS_FORMAT_KEY: ASSETS_FORMAT,
        ASSETS_SHA256_KEY: hashlib.sha256(payload).hexdigest(),
        ASSETS_FILE_COUNT_KEY: str(len(files)),
        ASSETS_ARCHIVE_KEY: base64.b64encode(payload).decode("ascii"),
    }


def _decode_assets(metadata: dict[str, str]) -> tuple[bytes, str]:
    archive_format = metadata.get(ASSETS_FORMAT_KEY)
    if archive_format != ASSETS_FORMAT:
        raise ValueError(
            f"Unsupported or missing embedded SenseNova assets format: {archive_format!r}. "
            "Reconvert the checkpoint with the current converter."
        )
    encoded = metadata.get(ASSETS_ARCHIVE_KEY)
    expected_sha256 = metadata.get(ASSETS_SHA256_KEY)
    if not encoded or not expected_sha256:
        raise ValueError("SenseNova checkpoint does not contain embedded config/tokenizer assets")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 in embedded SenseNova checkpoint assets") from exc
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Embedded SenseNova assets exceed {MAX_ARCHIVE_BYTES} bytes")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Embedded SenseNova checkpoint assets failed SHA256 verification")
    return payload, actual_sha256


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_ASSET_FILES:
        raise ValueError(f"Embedded checkpoint contains too many asset files: {len(members)}")
    total_size = 0
    for member in members:
        name = PurePosixPath(member.filename)
        if (
            member.is_dir()
            or "\\" in member.filename
            or name.is_absolute()
            or not name.parts
            or any(part in {"", ".", ".."} for part in name.parts)
        ):
            raise ValueError(f"Unsafe embedded checkpoint asset path: {member.filename!r}")
        total_size += member.file_size
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError(f"Embedded checkpoint assets exceed {MAX_EXTRACTED_BYTES} extracted bytes")
    return members


def _extract_assets(payload: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
        members = _safe_members(archive)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, mode="r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if not (destination / "config.json").is_file():
        raise ValueError("Embedded SenseNova checkpoint assets do not contain config.json")


def _checkpoint_identity(checkpoint: Path, assets_sha256: str) -> dict[str, object]:
    stat = checkpoint.stat()
    return {
        "format": ASSETS_FORMAT,
        "assets_sha256": assets_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
    }


def _cache_is_current(destination: Path, checkpoint: Path, identity: dict[str, object]) -> bool:
    manifest = destination / MANIFEST_NAME
    model_entry = destination / "model.safetensors"
    if not (destination / "config.json").is_file() or not manifest.is_file() or not model_entry.exists():
        return False
    try:
        cached_identity = json.loads(manifest.read_text(encoding="utf-8"))
        return cached_identity == identity and os.path.samefile(model_entry, checkpoint)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _link_checkpoint(checkpoint: Path, model_entry: Path) -> None:
    relative_target = os.path.relpath(checkpoint, model_entry.parent)
    try:
        model_entry.symlink_to(relative_target, target_is_directory=False)
        return
    except OSError:
        try:
            os.link(checkpoint, model_entry)
            return
        except OSError as hardlink_error:
            raise OSError(
                "Unable to create a symbolic link or hard link from the temporary SenseNova assets "
                f"directory to {checkpoint}. On Windows, enable Developer Mode or keep the checkpoint "
                "and ComfyUI temp directory on the same NTFS volume."
            ) from hardlink_error


def materialize_checkpoint_assets(
    checkpoint: Path,
    metadata: dict[str, str],
    temp_root: Path,
) -> Path:
    """Extract embedded assets to ``temp_root/NAME_assets`` and link the checkpoint."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SenseNova checkpoint not found: {checkpoint}")
    payload, assets_sha256 = _decode_assets(metadata)
    identity = _checkpoint_identity(checkpoint, assets_sha256)
    temp_root = temp_root.expanduser().resolve()
    destination = temp_root / f"{checkpoint.stem}_assets"

    with _CACHE_LOCK:
        temp_root.mkdir(parents=True, exist_ok=True)
        if _cache_is_current(destination, checkpoint, identity):
            return destination

        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=temp_root))
        try:
            _extract_assets(payload, staging)
            _link_checkpoint(checkpoint, staging / "model.safetensors")
            (staging / MANIFEST_NAME).write_text(
                json.dumps(identity, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if destination.exists() or destination.is_symlink():
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination
