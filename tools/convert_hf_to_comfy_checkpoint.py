#!/usr/bin/env python3
"""Convert a BF16 SenseNova Hugging Face snapshot to a ComfyUI checkpoint.

The output keeps the original Hugging Face tensor names so the plugin's pinned
Transformers 4.57.1 implementation remains the authority for model creation.
Alongside ``NAME.safetensors`` this tool creates ``NAME_assets/`` containing
the tokenizer/config files and a hard link named ``model.safetensors``.  The
hard link does not consume another copy of the model on filesystems that
support it.  Keep the checkpoint and its assets directory together.

No tensor is materialized: source safetensors byte ranges are copied directly,
so conversion needs only a small amount of RAM even for very large models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


PINNED_TRANSFORMERS = "4.57.1+SenseNova-patch"
PIXEL_VAE_KEY = "vae.pixel_space_vae"
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


@dataclass(frozen=True)
class TensorSlice:
    name: str
    dtype: str
    shape: list[int]
    source: Path | None
    source_offset: int
    length: int
    literal: bytes | None = None


def read_safetensors(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Not a safetensors file (short header): {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = handle.read(header_len)
    try:
        header = json.loads(header_raw.rstrip(b" \t\r\n\0"))
    except Exception as exc:
        raise ValueError(f"Invalid safetensors header: {path}") from exc
    return header, 8 + header_len


def source_files(repo: Path) -> list[Path]:
    index = repo / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(data.get("weight_map", {}).values()))
        files = [repo / name for name in names]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise FileNotFoundError(f"Hugging Face snapshot is incomplete; missing shards: {preview}{suffix}")
    elif (repo / "model.safetensors").is_file():
        files = [repo / "model.safetensors"]
    else:
        files = sorted(repo.glob("*.safetensors"))
    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No safetensors weights found in {repo}; convert PyTorch .bin weights to safetensors first."
        )
    return files


def collect_tensors(files: list[Path]) -> list[TensorSlice]:
    tensors: list[TensorSlice] = []
    seen: set[str] = set()
    for path in files:
        header, data_start = read_safetensors(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ValueError(f"Duplicate tensor {name!r} in {path}")
            seen.add(name)
            start, end = info["data_offsets"]
            tensors.append(
                TensorSlice(
                    name=name,
                    dtype=info["dtype"],
                    shape=list(info["shape"]),
                    source=path,
                    source_offset=data_start + int(start),
                    length=int(end) - int(start),
                )
            )
    if PIXEL_VAE_KEY in seen:
        raise ValueError(f"Source unexpectedly contains reserved key {PIXEL_VAE_KEY!r}")
    if not any(tensor.dtype == "BF16" for tensor in tensors):
        raise ValueError("Source contains no BF16 tensors; this converter only supports the original BF16 checkpoint.")
    tensors.append(
        TensorSlice(
            name=PIXEL_VAE_KEY,
            dtype="F32",
            shape=[],
            source=None,
            source_offset=0,
            length=4,
            literal=struct.pack("<f", 1.0),
        )
    )
    return tensors


def build_header(tensors: list[TensorSlice], assets_name: str, source: Path) -> bytes:
    offset = 0
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "comfyui_model_family": "sensenova_u1",
            "sensenova_checkpoint_version": "1",
            "sensenova_assets_dir": assets_name,
            "sensenova_transformers": PINNED_TRANSFORMERS,
            "sensenova_source": source.name,
            "sensenova_pixel_space_vae": "true",
        }
    }
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.length],
        }
        offset += tensor.length
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    # Safetensors requires the JSON header to be padded to an 8-byte boundary.
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def copy_range(source: Path, source_offset: int, length: int, output: BinaryIO) -> None:
    remaining = length
    with source.open("rb") as handle:
        handle.seek(source_offset)
        while remaining:
            chunk = handle.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"Unexpected EOF while reading {source}")
            output.write(chunk)
            remaining -= len(chunk)


def write_checkpoint(repo: Path, output: Path) -> None:
    files = source_files(repo)
    tensors = collect_tensors(files)
    assets_name = f"{output.stem}_assets"
    header = build_header(tensors, assets_name, repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(header)))
            handle.write(header)
            for index, tensor in enumerate(tensors, 1):
                print(f"[{index}/{len(tensors)}] {tensor.name}", file=sys.stderr)
                if tensor.literal is not None:
                    handle.write(tensor.literal)
                else:
                    assert tensor.source is not None
                    copy_range(tensor.source, tensor.source_offset, tensor.length, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def copy_assets(repo: Path, output: Path) -> Path:
    assets = output.with_name(f"{output.stem}_assets")
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    for path in repo.iterdir():
        if not path.is_file() or path.suffix.lower() in WEIGHT_SUFFIXES:
            continue
        if path.name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(path, assets / path.name)
    if not (assets / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {repo}")

    model_link = assets / "model.safetensors"
    relative = os.path.relpath(output, assets)
    try:
        model_link.symlink_to(relative)
    except OSError:
        try:
            os.link(output, model_link)
        except OSError:
            shutil.copy2(output, model_link)
            print("Warning: symlink/hard-link unavailable; assets contain a full checkpoint copy.", file=sys.stderr)
    return assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original BF16 Hugging Face snapshot directory")
    parser.add_argument("output", type=Path, help="Output .safetensors, normally ComfyUI/models/checkpoints/NAME.safetensors")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing checkpoint/assets directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".safetensors":
        raise ValueError("Output filename must end in .safetensors")
    assets = output.with_name(f"{output.stem}_assets")
    if (output.exists() or assets.exists()) and not args.overwrite:
        raise FileExistsError("Output or assets already exist; pass --overwrite to replace them")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Not a Hugging Face snapshot (missing config.json): {source}")
    write_checkpoint(source, output)
    assets = copy_assets(source, output)
    print(f"Checkpoint: {output}")
    print(f"Assets:     {assets}")
    print("Load it with the SenseNova Loader node; keep both paths together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
