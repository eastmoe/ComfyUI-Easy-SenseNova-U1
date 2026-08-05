from __future__ import annotations

import sys
from pathlib import Path

try:
    import folder_paths
except ImportError:  # 便于在 ComfyUI 之外执行轻量测试
    folder_paths = None


PLUGIN_DIR = Path(__file__).resolve().parents[2]
ORIGIN_DIR = PLUGIN_DIR / "origin" / "SenseNova-U1"
ORIGIN_SRC_DIR = ORIGIN_DIR / "src"
MODEL_FOLDER_KEY = "SenseNova"
MODEL_FOLDER_NAME = "SenseNova"


def comfy_root() -> Path:
    if folder_paths is not None and getattr(folder_paths, "base_path", None):
        return Path(folder_paths.base_path)
    return PLUGIN_DIR.parent.parent


def models_root() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    return comfy_root() / "models" / MODEL_FOLDER_NAME


def register_model_folder() -> None:
    """注册 ComfyUI/models/SenseNova 模型目录。"""
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    if folder_paths is None:
        return
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(MODEL_FOLDER_KEY, str(root), is_default=True)
    elif hasattr(folder_paths, "folder_names_and_paths"):
        current = folder_paths.folder_names_and_paths.get(MODEL_FOLDER_KEY)
        if current is None:
            folder_paths.folder_names_and_paths[MODEL_FOLDER_KEY] = ([str(root)], set())
        elif str(root) not in current[0]:
            current[0].append(str(root))


def ensure_origin_source() -> None:
    """让随节点分发的原项目源码可被 Python 导入。"""
    if not (ORIGIN_SRC_DIR / "sensenova_u1" / "__init__.py").is_file():
        raise RuntimeError(f"缺少 SenseNova-U1 后端源码: {ORIGIN_SRC_DIR}")
    source = str(ORIGIN_SRC_DIR)
    if source not in sys.path:
        sys.path.insert(0, source)


def available_models() -> list[str]:
    """列出包含 config.json 的模型子目录。"""
    root = models_root()
    if not root.is_dir():
        return []
    found = {str(path.parent.relative_to(root)) for path in root.rglob("config.json")}
    return sorted(name for name in found if name and name != ".")


def resolve_model_path(model_name: str, model_path: str = "") -> Path:
    """优先使用显式路径，否则解析模型下拉框中的相对子目录。"""
    text = (model_path or "").strip()
    if text and text.lower() not in {"auto", "自动"}:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = models_root() / candidate
    else:
        name = (model_name or "").strip()
        if not name or name == "<未找到模型>":
            choices = available_models()
            if not choices:
                raise RuntimeError(
                    f"在 {models_root()} 中未找到模型。请先运行模型下载节点，或填写模型路径。"
                )
            name = choices[0]
        candidate = models_root() / name
    candidate = candidate.resolve()
    root = models_root().resolve()
    if not candidate.is_relative_to(root):
        raise RuntimeError(f"模型路径必须位于 {root} 内: {candidate}")
    if not (candidate / "config.json").is_file():
        raise RuntimeError(f"模型目录无效（缺少 config.json）: {candidate}")
    return candidate
