from __future__ import annotations

import json
from pathlib import Path

from .src.comfy_easy_sensenova_u1 import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    register_model_folder,
)


PLUGIN_DIR = Path(__file__).resolve().parent


def _register_localization_route() -> None:
    """向前端提供节点简体中文本地化文件。"""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    @instance.routes.get("/easy_sensenova_u1/local/{locale}/nodes.json")
    async def get_nodes_localization(request):
        locale = request.match_info.get("locale", "zh-cn")
        path = PLUGIN_DIR / "local" / locale / "nodes.json"
        if not path.is_file():
            return web.json_response({}, status=404)
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response(data)


try:
    register_model_folder()
    _register_localization_route()
except Exception as exc:
    print(f"[Comfy-Easy-SenseNova-U1] 初始化失败: {exc}", flush=True)


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
