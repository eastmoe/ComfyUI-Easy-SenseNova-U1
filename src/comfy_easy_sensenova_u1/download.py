from __future__ import annotations

import inspect
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .paths import models_root


DOWNLOAD_LOCK = threading.RLock()
OFFICIAL_REPOS = {
    "SenseNova-U1-8B-MoT": "sensenova/SenseNova-U1-8B-MoT",
    "SenseNova-U1-A3B-MoT": "sensenova/SenseNova-U1-A3B-MoT",
    "SenseNova-U1-8B-MoT-Infographic": "sensenova/SenseNova-U1-8B-MoT-Infographic",
    "SenseNova-U1-8B-MoT-Infographic-V2": "sensenova/SenseNova-U1-8B-MoT-Infographic-V2",
    "SenseNova-U1-8B-MoT-Infographic-V3": "sensenova/SenseNova-U1-8B-MoT-Infographic-V3",
    "自定义仓库": "",
}


def _safe_subfolder(repo_id: str, subfolder: str) -> str:
    value = (subfolder or "").strip() or repo_id.rsplit("/", 1)[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not value:
        raise ValueError("模型子文件夹名称不能为空。")
    return value


@contextmanager
def _download_environment(
    endpoint: str | None,
    disable_tls: bool,
    disable_xet: bool,
) -> Iterator[None]:
    """临时设置下载网络选项，结束后完整恢复进程环境。"""
    old_endpoint = os.environ.get("HF_ENDPOINT")
    old_tls = os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION")
    old_xet = os.environ.get("HF_HUB_DISABLE_XET")
    original_requests = None
    original_httpx = None
    original_async_httpx = None
    hub_constants = None
    old_xet_constant = None

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
    if disable_tls:
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            from huggingface_hub import constants as hub_constants

            old_xet_constant = hub_constants.HF_HUB_DISABLE_XET
            hub_constants.HF_HUB_DISABLE_XET = True
        except (ImportError, AttributeError):
            hub_constants = None

    if disable_tls:
        try:
            import requests

            original_requests = requests.sessions.Session.request

            def request_without_verify(self, method, url, **kwargs):
                kwargs["verify"] = False
                return original_requests(self, method, url, **kwargs)

            requests.sessions.Session.request = request_without_verify
        except Exception:
            original_requests = None
        try:
            import httpx

            original_httpx = httpx.Client.__init__
            original_async_httpx = httpx.AsyncClient.__init__

            def httpx_without_verify(self, *args, **kwargs):
                kwargs["verify"] = False
                return original_httpx(self, *args, **kwargs)

            def async_httpx_without_verify(self, *args, **kwargs):
                kwargs["verify"] = False
                return original_async_httpx(self, *args, **kwargs)

            httpx.Client.__init__ = httpx_without_verify
            httpx.AsyncClient.__init__ = async_httpx_without_verify
        except Exception:
            original_httpx = original_async_httpx = None

    try:
        yield
    finally:
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint
        if old_tls is None:
            os.environ.pop("HF_HUB_DISABLE_SSL_VERIFICATION", None)
        else:
            os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = old_tls
        if old_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = old_xet
        if hub_constants is not None:
            hub_constants.HF_HUB_DISABLE_XET = old_xet_constant
        if original_requests is not None:
            import requests

            requests.sessions.Session.request = original_requests
        if original_httpx is not None and original_async_httpx is not None:
            import httpx

            httpx.Client.__init__ = original_httpx
            httpx.AsyncClient.__init__ = original_async_httpx


def download_snapshot(
    repo_id: str,
    subfolder: str,
    source: str,
    revision: str,
    token: str,
    disable_tls: bool,
    disable_xet: bool,
    force_download: bool,
) -> tuple[str, str]:
    """下载一个完整仓库快照到 models/SenseNova/独立子目录。"""
    repo_id = (repo_id or "").strip()
    if "/" not in repo_id:
        raise ValueError("仓库 ID 应为“组织/仓库”格式。")
    folder = _safe_subfolder(repo_id, subfolder)
    target = (models_root() / folder).resolve()
    root = models_root().resolve()
    if target.parent != root:
        raise ValueError("模型只能下载到 models/SenseNova 的直接子文件夹。")
    if (target / "config.json").is_file() and not force_download:
        return str(target), f"模型已存在，跳过下载: {target}"

    if source not in {"huggingface", "hfmirror"}:
        raise ValueError(f"不支持的下载源: {source}")
    # 显式传入 endpoint，避免进程启动前遗留的 HF_ENDPOINT 改写用户选择。
    endpoint = "https://hf-mirror.com" if source == "hfmirror" else "https://huggingface.co"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 huggingface_hub，请在 ComfyUI Python 环境安装 requirements.txt。") from exc

    target.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(target),
        "force_download": bool(force_download),
    }
    if revision.strip():
        kwargs["revision"] = revision.strip()
    if token.strip():
        kwargs["token"] = token.strip()
    signature = inspect.signature(snapshot_download)
    if endpoint and "endpoint" in signature.parameters:
        kwargs["endpoint"] = endpoint
    if "local_dir_use_symlinks" in signature.parameters:
        kwargs["local_dir_use_symlinks"] = False

    with DOWNLOAD_LOCK, _download_environment(endpoint, disable_tls, disable_xet):
        downloaded = snapshot_download(**kwargs)
    if not (target / "config.json").is_file():
        raise RuntimeError(f"下载结束，但目标目录缺少 config.json: {target}")
    return str(target), f"下载完成: {downloaded}"
