"""SenseNova-U1 专用的 Transformers 4.57.1 私有运行时。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


PATCH_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "transformers_4571"
PINNED_VERSION = "4.57.1"


def load_transformers() -> ModuleType:
    """加载插件内置版本，且不替换全局 ``transformers`` 模块。"""
    patch_path = str(PATCH_DIR)
    if patch_path not in sys.path:
        sys.path.insert(0, patch_path)

    module = importlib.import_module(PACKAGE_NAME)
    module_path = Path(module.__file__).resolve()
    expected_root = (PATCH_DIR / PACKAGE_NAME).resolve()
    if not module_path.is_relative_to(expected_root):
        raise RuntimeError(
            f"私有 Transformers 命名空间被其他模块占用: {module_path}"
        )
    if module.__version__ != PINNED_VERSION:
        raise RuntimeError(
            f"私有 Transformers 补丁版本错误: 期望 {PINNED_VERSION}，"
            f"实际 {module.__version__}。"
        )
    return module


__all__ = ["PACKAGE_NAME", "PATCH_DIR", "PINNED_VERSION", "load_transformers"]
