from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from typing import Any

import torch

from .download import FILE_VERIFICATIONS, OFFICIAL_REPOS, download_snapshot
from .paths import available_models, resolve_model_path
from .progress import (
    DiffusionInferenceProgress,
    ThinkingInferenceProgress,
    TokenInferenceProgress,
    throw_if_interrupted,
)
from .runtime import (
    ATTENTION_BACKENDS,
    CFG_NORMS,
    COMPUTE_PRECISIONS,
    DEFAULT_SEED,
    DEFAULT_SYSTEM_MESSAGE,
    DEVICE_MAPS,
    MODEL_TYPE,
    STORAGE_PRECISIONS,
    VRAM_MODES,
    SenseNovaHandle,
    available_devices,
    comfy_to_pil_batch,
    generated_to_comfy,
    load_handle,
    metadata,
    target_size_for_edit,
    validate_size,
)


CATEGORY = "eastmoe/Comfy-Easy-SenseNova-U1"


def ui(display_name: str, tooltip: str, **kwargs: Any) -> dict[str, Any]:
    return {"display_name": display_name, "tooltip": tooltip, **kwargs}


def common_sampling_inputs(include_image_cfg: bool = False) -> dict[str, Any]:
    values: dict[str, Any] = {
        "cfg_scale": ("FLOAT", ui("文本 CFG", "文本条件引导强度。", default=4.0, min=0.0, max=20.0, step=0.1)),
        "cfg_norm": (list(CFG_NORMS), ui("CFG 归一化", "CFG 重缩放方式；cfg_zero_star 仅用于文生图。")),
        "timestep_shift": ("FLOAT", ui("时间步偏移", "扩散时间步偏移，原项目推荐 3.0。", default=3.0, min=0.0, max=20.0, step=0.1)),
        "cfg_interval_start": ("FLOAT", ui("CFG 起点", "CFG 生效区间起点。", default=0.0, min=0.0, max=1.0, step=0.01)),
        "cfg_interval_end": ("FLOAT", ui("CFG 终点", "CFG 生效区间终点。", default=1.0, min=0.0, max=1.0, step=0.01)),
        "num_steps": ("INT", ui("采样步数", "扩散采样步数。", default=50, min=1, max=200, step=1)),
        "seed": ("INT", ui("随机种子", "固定种子可复现结果。", default=DEFAULT_SEED, min=0, max=0x7FFFFFFF)),
        "think_mode": ("BOOLEAN", ui("思考模式", "先生成推理文本，再完成图像任务。", default=True)),
    }
    if include_image_cfg:
        values = {
            "cfg_scale": values["cfg_scale"],
            "img_cfg_scale": ("FLOAT", ui("图像 CFG", "输入图像条件引导强度。", default=1.0, min=0.0, max=20.0, step=0.1)),
            **{key: value for key, value in values.items() if key != "cfg_scale"},
        }
    return values


class ComfyEasySenseNovaDownloadModel:
    """从 Hugging Face 或 hf-mirror 下载完整模型快照。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_preset": (list(OFFICIAL_REPOS), ui("模型预设", "选择官方模型；选择自定义后填写仓库 ID 和模型子文件夹。")),
                "repo_id": ("STRING", ui("仓库 ID", "仅自定义仓库时生效；选择官方预设时忽略。", default="sensenova/SenseNova-U1-8B-MoT")),
                "model_subfolder": ("STRING", ui("模型子文件夹", "仅自定义仓库时生效；官方预设自动使用仓库名。", default="")),
                "download_source": (["huggingface", "hfmirror"], ui("下载源", "选择 Hugging Face 官方站或 hf-mirror。")),
                "revision": ("STRING", ui("版本", "可选 branch、tag 或 commit；留空使用默认分支。", default="")),
                "token": ("STRING", ui("访问令牌", "私有/受限仓库令牌；留空使用本机已登录凭据。", default="", password=True)),
                "disable_tls": ("BOOLEAN", ui("关闭 TLS 校验", "仅在可信网络排障时关闭证书校验。", default=False)),
                "disable_xet": ("BOOLEAN", ui("关闭 Xet", "强制使用常规 HTTP 下载，镜像兼容性更好。", default=True)),
                "force_download": ("BOOLEAN", ui("强制重新下载", "关闭时自动复用完整文件并断点续传；开启后强制重新下载。", default=False)),
                "download_threads": ("INT", ui("并行下载线程", "同时下载的文件数；线程越多越占用网络、内存和磁盘 IO。", default=8, min=1, max=64, step=1)),
                "xet_connections": ("INT", ui("Xet 单文件连接数", "每个 Xet 文件的并发范围请求数；仅未关闭 Xet 时生效。", default=16, min=1, max=64, step=1)),
                "file_verification": (list(FILE_VERIFICATIONS), ui("文件校验", "大小校验较快；SHA256 会完整读取所有带远端哈希的权重文件。")),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("模型路径", "下载状态")
    FUNCTION = "download"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "将 SenseNova-U1 模型下载到 ComfyUI/models/SenseNova 的不同子文件夹；显示进度并支持停止与续传。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("force_download"):
            return float("nan")
        effective = dict(kwargs)
        if OFFICIAL_REPOS.get(effective.get("model_preset", "")):
            effective["repo_id"] = ""
            effective["model_subfolder"] = ""
        payload = json.dumps(effective, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def download(
        self,
        model_preset,
        repo_id,
        model_subfolder,
        download_source,
        revision,
        token,
        disable_tls,
        disable_xet,
        force_download,
        download_threads=8,
        xet_connections=16,
        file_verification="大小",
    ):
        preset_repo = OFFICIAL_REPOS.get(model_preset)
        resolved_repo = preset_repo or repo_id
        resolved_subfolder = "" if preset_repo else model_subfolder
        return download_snapshot(
            resolved_repo,
            resolved_subfolder,
            download_source,
            revision,
            token,
            disable_tls,
            disable_xet,
            force_download,
            download_threads,
            xet_connections,
            file_verification,
        )


class ComfyEasySenseNovaLoadModel:
    """加载并缓存模型，分别设置权重存储与推理计算精度。"""

    @classmethod
    def INPUT_TYPES(cls):
        choices = available_models() or ["<未找到模型>"]
        return {
            "required": {
                "model_name": (choices, ui("本地模型", "models/SenseNova 下检测到的模型子目录。")),
                "device": (available_devices(), ui("设备", "单设备目标；auto 使用 ComfyUI 当前计算设备。")),
                "storage_precision": (list(STORAGE_PRECISIONS), ui("存储精度", "模型权重加载并驻留时的数据类型。")),
                "compute_precision": (list(COMPUTE_PRECISIONS), ui("计算精度", "推理自动混合精度；auto 跟随存储精度。float32 计算要求 float32 存储。")),
                "attention_backend": (list(ATTENTION_BACKENDS), ui("注意力机制", "auto 自动选择；flash 需要 flash-attn；sdpa 使用 PyTorch SDPA。")),
                "vram_mode": (list(VRAM_MODES), ui("显存模式", "full 整模常驻；balanced 异步层预取；low 同步逐层卸载。")),
                "device_map": (list(DEVICE_MAPS), ui("多卡映射", "none 为单设备；其余值使用 Accelerate 分片。不能与层卸载同时启用。")),
                "max_memory": ("STRING", ui("设备内存上限", "device_map 的逐设备预算，例如 0=20GiB,1=20GiB,cpu=64GiB。", default="")),
                "reload_model": ("BOOLEAN", ui("重新加载模型", "忽略节点内部模型缓存。", default=False)),
            },
            "optional": {
                "model_path": ("STRING", ui("模型路径", "可连接下载节点输出；非空时优先于本地模型下拉框。", default="")),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("SenseNova 模型", "模型信息")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "从 models/SenseNova 加载模型，支持存储精度、计算精度、注意力机制、设备、多卡与低显存设置。"

    def load(self, model_name, device, storage_precision, compute_precision, attention_backend, vram_mode, device_map, max_memory, reload_model, model_path=""):
        resolved = resolve_model_path(model_name, model_path)
        handle = load_handle(
            str(resolved),
            device,
            storage_precision,
            compute_precision,
            attention_backend,
            vram_mode,
            device_map,
            max_memory,
            reload_model,
        )
        return handle, json.dumps(handle.info, ensure_ascii=False, indent=2)


class ComfyEasySenseNovaTextToImage:
    """SenseNova-U1 文生图（普通/思考模式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        sampling = common_sampling_inputs()
        return {
            "required": {
                "model": (MODEL_TYPE, ui("SenseNova 模型", "连接模型加载节点。")),
                "prompt": ("STRING", ui("提示词", "描述要生成的图像。", multiline=True, default="")),
                "width": ("INT", ui("图像宽度", "可自由设置，必须为 32 的倍数。", default=2048, min=32, max=8192, step=32)),
                "height": ("INT", ui("图像高度", "可自由设置，必须为 32 的倍数。", default=2048, min=32, max=8192, step=32)),
                **sampling,
                "batch_size": ("INT", ui("批量数量", "一次生成的图像数量。", default=1, min=1, max=16)),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图像", "思考文本", "元数据")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 SenseNova-U1 执行文生图，支持普通模式、思考模式、tqdm 进度与停止。"

    def generate(self, model: SenseNovaHandle, prompt, width, height, cfg_scale, cfg_norm, timestep_shift, cfg_interval_start, cfg_interval_end, num_steps, seed, think_mode, batch_size):
        if not prompt.strip():
            raise ValueError("提示词不能为空。")
        validate_size(width, height)
        if cfg_interval_start > cfg_interval_end:
            raise ValueError("CFG 起点不能大于 CFG 终点。")
        thinking_progress = (
            ThinkingInferenceProgress(model.model, 1024, "SenseNova 文生图思考", "token")
            if think_mode
            else nullcontext()
        )
        with model.generation_context() as backend, thinking_progress, DiffusionInferenceProgress(
            backend, num_steps, "SenseNova 文生图采样"
        ):
            result = backend.t2i_generate(
                model.tokenizer,
                prompt,
                image_size=(width, height),
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=(cfg_interval_start, cfg_interval_end),
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
            )
        tensor, think_text = result if think_mode else (result, "")
        return generated_to_comfy(tensor), think_text, metadata(model, "text_to_image", width=width, height=height, seed=seed, steps=num_steps)


class ComfyEasySenseNovaImageEdit:
    """使用一张或多张 ComfyUI 图像进行指令编辑。"""

    @classmethod
    def INPUT_TYPES(cls):
        sampling = common_sampling_inputs(include_image_cfg=True)
        sampling["cfg_norm"] = (list(CFG_NORMS[:-1]), ui("CFG 归一化", "图像编辑支持 none、global、channel。"))
        return {
            "required": {
                "model": (MODEL_TYPE, ui("SenseNova 模型", "连接模型加载节点。")),
                "image": ("IMAGE", ui("输入图像", "支持 ComfyUI 图像批次，批次中的图片作为多图参考。")),
                "prompt": ("STRING", ui("编辑指令", "说明需要修改的内容；未说明的属性应尽量保留。", multiline=True, default="")),
                "auto_size": ("BOOLEAN", ui("自动分辨率", "按首张输入图的宽高比与目标像素数计算输出尺寸。", default=True)),
                "width": ("INT", ui("输出宽度", "关闭自动分辨率时使用，必须为 32 的倍数。", default=2048, min=32, max=8192, step=32)),
                "height": ("INT", ui("输出高度", "关闭自动分辨率时使用，必须为 32 的倍数。", default=2048, min=32, max=8192, step=32)),
                "target_megapixels": ("FLOAT", ui("目标百万像素", "自动分辨率的总像素预算。", default=4.194304, min=0.25, max=32.0, step=0.25)),
                **sampling,
                "batch_size": ("INT", ui("输出批量数量", "一次生成的编辑结果数量。", default=1, min=1, max=16)),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("编辑图像", "思考文本", "元数据")
    FUNCTION = "edit"
    CATEGORY = CATEGORY
    DESCRIPTION = "SenseNova-U1 图像编辑，可把 IMAGE 批次作为多张参考图，并显示可停止的采样进度。"

    def edit(self, model: SenseNovaHandle, image: torch.Tensor, prompt, auto_size, width, height, target_megapixels, cfg_scale, img_cfg_scale, cfg_norm, timestep_shift, cfg_interval_start, cfg_interval_end, num_steps, seed, think_mode, batch_size):
        if not prompt.strip():
            raise ValueError("编辑指令不能为空。")
        images = comfy_to_pil_batch(image)
        if auto_size:
            width, height = target_size_for_edit(images[0], target_megapixels)
        validate_size(width, height)
        if cfg_interval_start > cfg_interval_end:
            raise ValueError("CFG 起点不能大于 CFG 终点。")
        thinking_progress = (
            ThinkingInferenceProgress(model.model, 1024, "SenseNova 图像编辑思考", "token")
            if think_mode
            else nullcontext()
        )
        with model.generation_context() as backend, thinking_progress, DiffusionInferenceProgress(
            backend, num_steps, "SenseNova 图像编辑采样"
        ):
            result = backend.it2i_generate(
                model.tokenizer,
                prompt,
                images,
                image_size=(width, height),
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=(cfg_interval_start, cfg_interval_end),
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
            )
        tensor, think_text = result if think_mode else (result, "")
        return generated_to_comfy(tensor), think_text, metadata(model, "image_edit", width=width, height=height, seed=seed, input_images=len(images), steps=num_steps)


class ComfyEasySenseNovaVisionQA:
    """视觉理解 / VQA 节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE, ui("SenseNova 模型", "连接模型加载节点。")),
                "image": ("IMAGE", ui("输入图像", "待理解或问答的图像；支持批次。")),
                "question": ("STRING", ui("问题", "对图像提出的问题或描述要求。", multiline=True, default="描述这张图片。")),
                "max_new_tokens": ("INT", ui("最大新 Token", "回答最多生成的 token 数。", default=1024, min=1, max=8192)),
                "do_sample": ("BOOLEAN", ui("启用采样", "关闭时使用贪心解码。", default=False)),
                "temperature": ("FLOAT", ui("温度", "采样随机性。", default=0.7, min=0.01, max=2.0, step=0.01)),
                "top_p": ("FLOAT", ui("Top P", "核采样概率阈值。", default=0.9, min=0.01, max=1.0, step=0.01)),
                "top_k": ("INT", ui("Top K", "0 表示不显式设置 Top K。", default=0, min=0, max=4096)),
                "repetition_penalty": ("FLOAT", ui("重复惩罚", "1.0 表示不额外惩罚。", default=1.0, min=0.1, max=4.0, step=0.01)),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("回答", "回答列表 JSON", "元数据")
    FUNCTION = "answer"
    CATEGORY = CATEGORY
    DESCRIPTION = "SenseNova-U1 视觉理解与视觉问答，图像批次会逐张回答，并显示可停止的 token 进度。"

    def answer(self, model: SenseNovaHandle, image: torch.Tensor, question, max_new_tokens, do_sample, temperature, top_p, top_k, repetition_penalty):
        if not question.strip():
            raise ValueError("问题不能为空。")
        from sensenova_u1.models.neo_unify.utils import load_image_native

        config: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            config.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                config["top_k"] = top_k
        if repetition_penalty != 1.0:
            config["repetition_penalty"] = repetition_penalty
        input_images = comfy_to_pil_batch(image)
        answers = []
        with TokenInferenceProgress(
            model.model,
            max_new_tokens * len(input_images),
            "SenseNova 视觉问答推理",
            "token",
        ) as progress:
            for pil_image in input_images:
                throw_if_interrupted()
                pixel_values, grid_hw = load_image_native(pil_image)
                pixel_values = pixel_values.to(model.input_device, dtype=model.model.dtype)
                grid_hw = grid_hw.to(model.input_device)
                request_config = config.copy()
                request_config["stopping_criteria"] = progress.stopping_criteria()
                with model.generation_context() as backend:
                    response, _ = backend.chat(
                        model.tokenizer,
                        pixel_values,
                        question,
                        request_config,
                        history=None,
                        return_history=True,
                        grid_hw=grid_hw,
                    )
                answers.append(response)
        joined = answers[0] if len(answers) == 1 else "\n\n".join(f"[{i + 1}] {answer}" for i, answer in enumerate(answers))
        return joined, json.dumps(answers, ensure_ascii=False, indent=2), metadata(model, "vision_qa", image_count=len(answers), max_new_tokens=max_new_tokens)


class ComfyEasySenseNovaInterleave:
    """原生图文交错生成，可选输入参考图。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE, ui("SenseNova 模型", "连接模型加载节点。")),
                "prompt": ("STRING", ui("提示词", "描述需要生成的图文内容；可包含多步任务。", multiline=True, default="")),
                "width": ("INT", ui("图像宽度", "每张生成图像的宽度，必须为 32 的倍数。", default=1536, min=32, max=8192, step=32)),
                "height": ("INT", ui("图像高度", "每张生成图像的高度，必须为 32 的倍数。", default=1536, min=32, max=8192, step=32)),
                "system_message": ("STRING", ui("系统提示词", "约束思考、文本与图像交错格式。", multiline=True, default=DEFAULT_SYSTEM_MESSAGE)),
                "cfg_scale": ("FLOAT", ui("文本 CFG", "文本条件引导强度。", default=4.0, min=0.0, max=20.0, step=0.1)),
                "img_cfg_scale": ("FLOAT", ui("图像 CFG", "输入图像条件引导强度。", default=1.0, min=0.0, max=20.0, step=0.1)),
                "timestep_shift": ("FLOAT", ui("时间步偏移", "扩散时间步偏移。", default=3.0, min=0.0, max=20.0, step=0.1)),
                "cfg_interval_start": ("FLOAT", ui("CFG 起点", "CFG 生效区间起点。", default=0.0, min=0.0, max=1.0, step=0.01)),
                "cfg_interval_end": ("FLOAT", ui("CFG 终点", "CFG 生效区间终点。", default=1.0, min=0.0, max=1.0, step=0.01)),
                "num_steps": ("INT", ui("每张图采样步数", "每张生成图像使用的扩散步数。", default=50, min=1, max=200)),
                "max_images": ("INT", ui("最大图像数", "一次交错回答最多生成的图像数量。", default=10, min=1, max=32)),
                "seed": ("INT", ui("随机种子", "固定种子可复现结果。", default=DEFAULT_SEED, min=0, max=0x7FFFFFFF)),
                "think_mode": ("BOOLEAN", ui("思考模式", "启用模型的原生图文交错推理。", default=True)),
            },
            "optional": {
                "image": ("IMAGE", ui("参考图像", "可选；IMAGE 批次中的图片全部作为输入参考。")),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("生成图像", "交错文本", "元数据")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "SenseNova-U1 原生图文交错生成，输出文本中的 <image> 与图像批次按顺序对应；显示可停止的采样进度。"

    def generate(self, model: SenseNovaHandle, prompt, width, height, system_message, cfg_scale, img_cfg_scale, timestep_shift, cfg_interval_start, cfg_interval_end, num_steps, max_images, seed, think_mode, image=None):
        if not prompt.strip():
            raise ValueError("提示词不能为空。")
        validate_size(width, height)
        input_images = comfy_to_pil_batch(image) if image is not None else []
        if cfg_interval_start > cfg_interval_end:
            raise ValueError("CFG 起点不能大于 CFG 终点。")
        thinking_progress = (
            ThinkingInferenceProgress(model.model, 8192, "SenseNova 图文交错思考/文本", "token")
            if think_mode
            else nullcontext()
        )
        with model.generation_context() as backend, thinking_progress, DiffusionInferenceProgress(
            backend, num_steps * max_images, "SenseNova 图文交错采样"
        ):
            text, image_tensors = backend.interleave_gen(
                model.tokenizer,
                prompt,
                images=input_images,
                image_size=(width, height),
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                timestep_shift=timestep_shift,
                cfg_interval=(cfg_interval_start, cfg_interval_end),
                num_steps=num_steps,
                max_images=max_images,
                system_message=system_message,
                think_mode=think_mode,
                seed=seed,
            )
        generated = []
        for tensor in image_tensors:
            throw_if_interrupted()
            batch = tensor if tensor.ndim == 4 else tensor.unsqueeze(0)
            generated.append(generated_to_comfy(batch)[0])
        output = torch.stack(generated) if generated else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return output, text, metadata(model, "interleave", width=width, height=height, seed=seed, input_images=len(input_images), output_images=len(generated), steps=num_steps)


NODE_CLASS_MAPPINGS = {
    "ComfyEasySenseNovaDownloadModel": ComfyEasySenseNovaDownloadModel,
    "ComfyEasySenseNovaLoadModel": ComfyEasySenseNovaLoadModel,
    "ComfyEasySenseNovaTextToImage": ComfyEasySenseNovaTextToImage,
    "ComfyEasySenseNovaImageEdit": ComfyEasySenseNovaImageEdit,
    "ComfyEasySenseNovaVisionQA": ComfyEasySenseNovaVisionQA,
    "ComfyEasySenseNovaInterleave": ComfyEasySenseNovaInterleave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyEasySenseNovaDownloadModel": "SenseNova-U1 模型下载",
    "ComfyEasySenseNovaLoadModel": "SenseNova-U1 模型加载",
    "ComfyEasySenseNovaTextToImage": "SenseNova-U1 文生图",
    "ComfyEasySenseNovaImageEdit": "SenseNova-U1 图像编辑",
    "ComfyEasySenseNovaVisionQA": "SenseNova-U1 视觉问答",
    "ComfyEasySenseNovaInterleave": "SenseNova-U1 图文交错生成",
}
