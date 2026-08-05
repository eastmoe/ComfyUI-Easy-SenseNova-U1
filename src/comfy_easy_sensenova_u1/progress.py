from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from tqdm.auto import tqdm


def throw_if_interrupted() -> None:
    """使用 ComfyUI 的标准中断异常响应队列停止按钮。"""
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    checker = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if checker is not None:
        checker()


class ComfyInferenceTqdm(tqdm):
    """同时显示终端 tqdm 与 ComfyUI 节点进度。"""

    def __init__(self, *args: Any, **kwargs: Any):
        self._comfy_progress = None
        super().__init__(*args, **kwargs)
        self._sync_progress()

    def _sync_progress(self) -> None:
        total = int(self.total or 0)
        if total <= 0:
            return
        if self._comfy_progress is None:
            try:
                import comfy.utils

                self._comfy_progress = comfy.utils.ProgressBar(total)
            except Exception:
                return
        self._comfy_progress.update_absolute(min(int(self.n), total), total)

    def update(self, n: int | float = 1):
        throw_if_interrupted()
        displayed = super().update(n)
        self._sync_progress()
        return displayed

    def refresh(self, *args: Any, **kwargs: Any):
        displayed = super().refresh(*args, **kwargs)
        self._sync_progress()
        return displayed


class _ModelProgress(AbstractContextManager):
    def __init__(self, model: Any, total: int, description: str, unit: str = "步"):
        self.model = model
        self.bar = ComfyInferenceTqdm(
            total=max(int(total), 1),
            desc=description,
            unit=unit,
            dynamic_ncols=True,
        )
        self._hooks: list[Any] = []

    def __enter__(self):
        throw_if_interrupted()
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None and hasattr(language_model, "register_forward_pre_hook"):
            self._hooks.append(language_model.register_forward_pre_hook(self._check_forward))
        return self

    def _check_forward(self, _module: Any, _args: Any) -> None:
        # 思考模式和逐 token 解码可能不会触发扩散进度，但仍需及时响应停止。
        throw_if_interrupted()

    def step(self) -> None:
        self.bar.update(1)

    def __exit__(self, exc_type, exc, tb):
        for hook in reversed(self._hooks):
            hook.remove()
        self._hooks.clear()
        try:
            if exc_type is None:
                throw_if_interrupted()
                # 生成可能因 EOS 或少于 max_images 提前结束，以实际工作量结束进度条。
                actual = max(int(self.bar.n), 1)
                self.bar.total = actual
                self.bar.n = actual
                self.bar.refresh()
                self.bar._sync_progress()
        finally:
            self.bar.close()
        return False


class DiffusionInferenceProgress(_ModelProgress):
    """以时间步嵌入模块的真实调用次数跟踪扩散采样。"""

    def __enter__(self):
        super().__enter__()
        modules = getattr(self.model, "fm_modules", None)
        timestep_embedder = None
        if modules is not None:
            try:
                timestep_embedder = modules["timestep_embedder"]
            except (KeyError, TypeError):
                pass
        if timestep_embedder is None or not hasattr(timestep_embedder, "register_forward_pre_hook"):
            self.__exit__(RuntimeError, None, None)
            raise RuntimeError("当前 SenseNova 模型缺少时间步模块，无法显示扩散推理进度。")
        self._hooks.append(timestep_embedder.register_forward_pre_hook(self._sample_step))
        return self

    def _sample_step(self, _module: Any, _args: Any) -> None:
        self.step()


class TokenInferenceProgress(_ModelProgress):
    """通过 Transformers StoppingCriteria 跟踪逐 token 推理。"""

    def stopping_criteria(self):
        from transformers import StoppingCriteria, StoppingCriteriaList

        progress = self

        class _ProgressAndInterruptCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                progress.step()
                return False

        return StoppingCriteriaList([_ProgressAndInterruptCriteria()])
