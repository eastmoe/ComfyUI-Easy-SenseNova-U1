from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

from .paths import models_root
from .progress import throw_if_interrupted


DOWNLOAD_LOCK = threading.RLock()
FILE_VERIFICATIONS = ("大小", "大小和 SHA256", "关闭")
OFFICIAL_REPOS = {
    "SenseNova-U1.5-8B-MoT": "sensenova/SenseNova-U1.5-8B-MoT",
    "SenseNova-U1.5-8B-MoT-Preview": "sensenova/SenseNova-U1.5-8B-MoT-Preview",
    "SenseNova-U1-8B-MoT": "sensenova/SenseNova-U1-8B-MoT",
    "SenseNova-U1-A3B-MoT": "sensenova/SenseNova-U1-A3B-MoT",
    "SenseNova-U1-8B-MoT-Infographic": "sensenova/SenseNova-U1-8B-MoT-Infographic",
    "SenseNova-U1-8B-MoT-Infographic-V2": "sensenova/SenseNova-U1-8B-MoT-Infographic-V2",
    "SenseNova-U1-8B-MoT-Infographic-V3": "sensenova/SenseNova-U1-8B-MoT-Infographic-V3",
    "自定义仓库": "",
}


class ComfyDownloadProgress(tqdm):
    """保留终端速度显示，并将总体字节进度同步到 ComfyUI。"""

    def __init__(self, *args, **kwargs):
        self._comfy_progress = None
        description = str(kwargs.get("desc", ""))
        self._sync_to_comfy = description.startswith(("Reconstructing", "校验文件"))
        if description.startswith("Downloading bytes"):
            kwargs["bar_format"] = (
                "{desc}: {bar}| {n_fmt:>5}B [{elapsed}, {rate_fmt}]{postfix}"
            )
        elif description.startswith("Reconstructing"):
            kwargs["bar_format"] = (
                "{l_bar}{bar}| {n_fmt:>5}B / {total_fmt:>5}B "
                "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
            )
        super().__init__(*args, **kwargs)
        self._sync_progress()

    def _sync_progress(self) -> None:
        total = int(self.total or 0)
        if not self._sync_to_comfy or total <= 0:
            return
        if self._comfy_progress is None:
            try:
                import comfy.utils

                self._comfy_progress = comfy.utils.ProgressBar(total)
            except Exception:
                self._sync_to_comfy = False
                return
        self._comfy_progress.update_absolute(min(int(self.n), total), total)

    def update(self, n=1):
        throw_if_interrupted()
        displayed = super().update(n)
        self._sync_progress()
        return displayed

    def refresh(self, *args, **kwargs):
        displayed = super().refresh(*args, **kwargs)
        self._sync_progress()
        return displayed


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
    xet_connections: int,
) -> Iterator[None]:
    """临时设置下载网络选项，结束后完整恢复进程环境。"""
    old_endpoint = os.environ.get("HF_ENDPOINT")
    old_tls = os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION")
    old_xet = os.environ.get("HF_HUB_DISABLE_XET")
    old_xet_connections = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS")
    original_requests = None
    original_httpx = None
    original_async_httpx = None
    hub_constants = None
    old_xet_constant = None
    reset_xet_session = None

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
    if disable_tls:
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    else:
        os.environ.pop("HF_HUB_DISABLE_SSL_VERIFICATION", None)
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    else:
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(xet_connections)
    try:
        from huggingface_hub import constants as hub_constants

        old_xet_constant = hub_constants.HF_HUB_DISABLE_XET
        hub_constants.HF_HUB_DISABLE_XET = bool(disable_xet)
    except (ImportError, AttributeError):
        hub_constants = None
    if not disable_xet:
        try:
            from huggingface_hub.utils._xet import abort_xet_session

            reset_xet_session = abort_xet_session
            # XetSession 在构造时读取并发设置；清理旧会话以应用本次节点参数。
            reset_xet_session()
        except (ImportError, AttributeError):
            reset_xet_session = None

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
        if old_xet_connections is None:
            os.environ.pop("HF_XET_NUM_CONCURRENT_RANGE_GETS", None)
        else:
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = old_xet_connections
        if hub_constants is not None:
            hub_constants.HF_HUB_DISABLE_XET = old_xet_constant
        if reset_xet_session is not None:
            # 避免本节点的临时连接数影响进程内其他 Hugging Face 下载。
            reset_xet_session()
        if original_requests is not None:
            import requests

            requests.sessions.Session.request = original_requests
        if original_httpx is not None and original_async_httpx is not None:
            import httpx

            httpx.Client.__init__ = original_httpx
            httpx.AsyncClient.__init__ = original_async_httpx


@contextmanager
def _quiet_http_request_logs() -> Iterator[None]:
    """隐藏 httpx/httpcore 的逐请求 INFO URL，保留警告与下载进度。"""
    loggers = [logging.getLogger(name) for name in ("httpx", "httpcore")]
    old_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.WARNING)
        yield
    finally:
        for logger, level in zip(loggers, old_levels):
            logger.setLevel(level)


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def _sha256_file(path: Path, progress: ComfyDownloadProgress) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            throw_if_interrupted()
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def verify_snapshot(
    target: Path,
    repo_id: str,
    revision: str,
    token: str,
    endpoint: str,
    verification: str,
) -> str:
    """按 Hub 远端清单检查缺失文件、文件大小及可用的 LFS SHA256。"""
    if verification == "关闭":
        if not (target / "config.json").is_file():
            raise RuntimeError(f"下载结束，但目标目录缺少 config.json: {target}")
        return "已关闭额外校验"
    if verification not in FILE_VERIFICATIONS:
        raise ValueError(f"不支持的文件校验方式: {verification}")

    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint, token=token.strip() or None)
    info = api.model_info(
        repo_id,
        revision=revision.strip() or None,
        files_metadata=True,
    )
    siblings = list(info.siblings or [])
    failures: list[str] = []
    checked_sizes = 0
    hash_entries: list[tuple[Path, str, str]] = []

    for sibling in siblings:
        throw_if_interrupted()
        relative = str(sibling.rfilename)
        path = (target / relative).resolve()
        if not path.is_relative_to(target):
            failures.append(f"非法远端路径: {relative}")
            continue
        if not path.is_file():
            failures.append(f"缺少文件: {relative}")
            continue
        expected_size = getattr(sibling, "size", None)
        if expected_size is not None:
            checked_sizes += 1
            actual_size = path.stat().st_size
            if actual_size != int(expected_size):
                failures.append(
                    f"大小不符: {relative}（本地 {actual_size}，远端 {expected_size}）"
                )
                continue
        expected_hash = _lfs_sha256(sibling)
        if verification == "大小和 SHA256" and expected_hash:
            hash_entries.append((path, relative, expected_hash.lower()))

    if failures:
        details = "\n".join(failures[:20])
        more = f"\n另有 {len(failures) - 20} 项错误。" if len(failures) > 20 else ""
        raise RuntimeError(f"模型文件校验失败：\n{details}{more}\n请重新运行下载节点续传或强制下载。")

    if hash_entries:
        total_bytes = sum(path.stat().st_size for path, _, _ in hash_entries)
        with ComfyDownloadProgress(
            total=total_bytes,
            desc="校验文件 SHA256",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for path, relative, expected_hash in hash_entries:
                throw_if_interrupted()
                actual_hash = _sha256_file(path, progress)
                if actual_hash != expected_hash:
                    failures.append(f"SHA256 不符: {relative}")
        if failures:
            details = "\n".join(failures[:20])
            raise RuntimeError(f"模型文件校验失败：\n{details}\n请启用强制下载后重试。")

    return (
        f"校验通过：远端 {len(siblings)} 个文件，"
        f"大小 {checked_sizes} 个，SHA256 {len(hash_entries)} 个"
    )


@contextmanager
def _interruptible_download_lock() -> Iterator[None]:
    """等待其他下载任务时也保持停止按钮可响应。"""
    acquired = False
    try:
        while not acquired:
            throw_if_interrupted()
            acquired = DOWNLOAD_LOCK.acquire(timeout=0.2)
        yield
    finally:
        if acquired:
            DOWNLOAD_LOCK.release()


def download_snapshot(
    repo_id: str,
    subfolder: str,
    source: str,
    revision: str,
    token: str,
    disable_tls: bool,
    disable_xet: bool,
    force_download: bool,
    download_threads: int,
    xet_connections: int,
    file_verification: str,
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
    download_threads = int(download_threads)
    xet_connections = int(xet_connections)
    if not 1 <= download_threads <= 64:
        raise ValueError("并行下载线程数必须在 1～64 之间。")
    if not 1 <= xet_connections <= 64:
        raise ValueError("Xet 单文件连接数必须在 1～64 之间。")

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
        "max_workers": download_threads,
        "tqdm_class": ComfyDownloadProgress,
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

    throw_if_interrupted()
    with (
        _interruptible_download_lock(),
        _download_environment(endpoint, disable_tls, disable_xet, xet_connections),
        _quiet_http_request_logs(),
    ):
        # force_download=False 时，Hub 会复用已完成文件并从 .incomplete 文件续传。
        downloaded = snapshot_download(**kwargs)
        throw_if_interrupted()
        verification_status = verify_snapshot(
            target,
            repo_id,
            revision,
            token,
            endpoint,
            file_verification,
        )
        throw_if_interrupted()
    return str(target), f"下载/续传完成: {downloaded}\n{verification_status}"
