"""ComfyUI-native sampling adapter for SenseNova-U1.

The adapter deliberately keeps the original model and tokenizer code.  It only
translates SenseNova's pixel-space rectified-flow contract into ComfyUI's MODEL,
CONDITIONING, VAE and GUIDER contracts.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.samplers
import comfy.sd
import comfy.supported_models_base

from .runtime import SenseNovaHandle, comfy_to_pil_batch


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class SenseNovaModelConfig(comfy.supported_models_base.BASE):
    unet_config = {"disable_unet_model_creation": True}
    latent_format = comfy.latent_formats.HiDreamO1Pixel
    sampling_settings = {"shift": 3.0, "noise_scale": 1.0, "multiplier": 1000}
    memory_usage_factor = 1.0

    def inpaint_model(self):
        return False


class SenseNovaCondition(comfy.conds.CONDConstant):
    """A non-concatenable cache selector.

    DynamicCache instances from the pinned Transformers fork must never be
    merged by ComfyUI's conditional batching code.
    """

    def can_concat(self, other):
        return False


@dataclass
class PreparedBranch:
    cache: Any
    indexes: torch.Tensor


@dataclass
class SenseNovaConditionBundle:
    prompt: str
    images: list[Any]
    think_mode: bool
    max_think_tokens: int
    think_text: str = ""
    prepared: dict[str, PreparedBranch] = field(default_factory=dict)
    prepared_shape: tuple[int, int, int] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def clear(self) -> None:
        if self.prepared:
            from sensenova_u1.models.neo_unify.modeling_neo_chat import clear_flash_kv_cache

            for branch in self.prepared.values():
                clear_flash_kv_cache(branch.cache)
        self.prepared.clear()
        self.prepared_shape = None


@dataclass(frozen=True)
class SenseNovaBranchSpec:
    bundle: SenseNovaConditionBundle
    branch: str


def conditioning_from_prompt(
    prompt: str,
    image: torch.Tensor | None,
    think_mode: bool,
    max_think_tokens: int,
) -> tuple[list, list, list, SenseNovaConditionBundle]:
    images = comfy_to_pil_batch(image) if image is not None else []
    if images and prompt.count("<image>") > len(images):
        raise ValueError("提示词中的 <image> 数量不能超过参考图像数量。")
    bundle = SenseNovaConditionBundle(
        prompt=prompt,
        images=images,
        think_mode=think_mode,
        max_think_tokens=max_think_tokens,
    )

    def make(branch: str) -> list:
        # The dummy tensor satisfies CONDITIONING's outer container contract;
        # extra_conds intentionally ignores cross_attn and uses the branch spec.
        return [[torch.zeros((1, 1, 1)), {"sensenova_spec": SenseNovaBranchSpec(bundle, branch)}]]

    positive = make("positive")
    middle = make("image") if images else make("negative")
    negative = make("negative")
    return positive, middle, negative, bundle


def _expand_cache(cache, batch_size: int) -> None:
    if cache is None:
        return
    for layer in cache.layers:
        layer.keys = layer.keys.expand(batch_size, *layer.keys.shape[1:])
        layer.values = layer.values.expand(batch_size, *layer.values.shape[1:])


def _prepare_text_bundle(wrapper: "SenseNovaComfyModel", bundle: SenseNovaConditionBundle, width: int, height: int, batch: int) -> None:
    model = wrapper.diffusion_model
    tokenizer = wrapper.tokenizer
    from sensenova_u1.models.neo_unify.modeling_neo_chat import prepare_flash_kv_cache
    from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

    merge_size = int(1 / model.downsample_ratio)
    token_h = height // (model.patch_size * merge_size)
    token_w = width // (model.patch_size * merge_size)
    think_content = "<think>\n" if bundle.think_mode else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
    query = model._build_t2i_query(bundle.prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
    uncond_query = model._build_t2i_query("", append_text=IMG_START_TOKEN)
    ids, indexes, mask = model._build_t2i_text_inputs(tokenizer, query)
    uncond_ids, uncond_indexes, uncond_mask = model._build_t2i_text_inputs(tokenizer, uncond_query)

    positive_indexes = model._build_t2i_image_indexes(token_h, token_w, indexes.shape[1], device=ids.device)
    if bundle.think_mode:
        outputs = model.language_model(
            input_ids=ids,
            indexes=indexes,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )
        positive_cache = outputs.past_key_values
        t_index = indexes[0].max().item()
        positive_cache, t_index, bundle.think_text = model._generate_think(
            tokenizer,
            outputs,
            positive_cache,
            t_index,
            IMG_START_TOKEN,
            max_think_tokens=bundle.max_think_tokens,
        )
        positive_indexes = model._build_t2i_image_indexes(token_h, token_w, t_index + 1, device=ids.device)
    else:
        positive_cache, _ = model._t2i_prefix_forward(ids, indexes, mask)
    negative_cache, _ = model._t2i_prefix_forward(uncond_ids, uncond_indexes, uncond_mask)
    negative_indexes = model._build_t2i_image_indexes(
        token_h, token_w, uncond_indexes.shape[1], device=uncond_ids.device
    )
    _expand_cache(positive_cache, batch)
    _expand_cache(negative_cache, batch)
    prepare_flash_kv_cache(positive_cache, current_len=token_h * token_w, batch_size=batch)
    prepare_flash_kv_cache(negative_cache, current_len=token_h * token_w, batch_size=batch)
    bundle.prepared = {
        "positive": PreparedBranch(positive_cache, positive_indexes),
        "image": PreparedBranch(negative_cache, negative_indexes),
        "negative": PreparedBranch(negative_cache, negative_indexes),
    }


def _prepare_edit_bundle(wrapper: "SenseNovaComfyModel", bundle: SenseNovaConditionBundle, width: int, height: int, batch: int) -> None:
    model = wrapper.diffusion_model
    tokenizer = wrapper.tokenizer
    from sensenova_u1.models.neo_unify.modeling_neo_chat import prepare_flash_kv_cache
    from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN, load_image_native

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    prompt = bundle.prompt
    image_count = prompt.count("<image>")
    if len(bundle.images) > image_count:
        if image_count == 0 and len(bundle.images) > 1:
            prompt = "".join(f"Image-{i + 1}:<image>\n" for i in range(len(bundle.images))) + prompt
        else:
            prompt = "<image>\n" * (len(bundle.images) - image_count) + prompt

    pixels, grids = [], []
    for image in bundle.images:
        pixel, grid = load_image_native(
            image,
            model.patch_size,
            model.downsample_ratio,
            min_pixels=512 * 512,
            max_pixels=min(2048 * 2048, (4096 * 4096) // len(bundle.images)),
            upscale=False,
        )
        pixels.append(pixel.to(model.device, dtype=torch.bfloat16))
        grids.append(grid.to(model.device))
    pixel_values = torch.cat(pixels)
    grid_hw = torch.cat(grids)

    think_content = "<think>\n" if bundle.think_mode else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
    queries = {
        "positive": model._build_t2i_query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content),
        "image": model._build_t2i_query("<image>" * len(bundle.images), append_text=IMG_START_TOKEN),
        "negative": model._build_t2i_query("", append_text=IMG_START_TOKEN),
    }
    for i in range(grid_hw.shape[0]):
        token_count = int(grid_hw[i, 0] * grid_hw[i, 1] * model.downsample_ratio**2)
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
        queries["positive"] = queries["positive"].replace("<image>", image_tokens, 1)
        queries["image"] = queries["image"].replace("<image>", image_tokens, 1)

    inputs: dict[str, tuple] = {}
    for name, query in queries.items():
        if name == "negative":
            inputs[name] = model._build_it2i_inputs(tokenizer, query)
        else:
            inputs[name] = model._build_it2i_inputs(tokenizer, query, pixel_values, grid_hw)

    merge_size = int(1 / model.downsample_ratio)
    token_h = height // (model.patch_size * merge_size)
    token_w = width // (model.patch_size * merge_size)
    prepared: dict[str, PreparedBranch] = {}
    for name, (embeds, indexes, mask) in inputs.items():
        image_indexes = model._build_t2i_image_indexes(
            token_h, token_w, indexes[0].max() + 1, device=embeds.device
        )
        if name == "positive" and bundle.think_mode:
            outputs = model.language_model(
                inputs_embeds=embeds,
                indexes=indexes,
                attention_mask=mask,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = outputs.past_key_values
            t_index = indexes[0].max().item()
            cache, t_index, bundle.think_text = model._generate_think(
                tokenizer,
                outputs,
                cache,
                t_index,
                IMG_START_TOKEN,
                max_think_tokens=bundle.max_think_tokens,
            )
            image_indexes = model._build_t2i_image_indexes(
                token_h, token_w, t_index + 1, device=embeds.device
            )
        else:
            cache, _ = model._it2i_prefix_forward(embeds, indexes, mask)
        _expand_cache(cache, batch)
        prepare_flash_kv_cache(cache, current_len=token_h * token_w, batch_size=batch)
        prepared[name] = PreparedBranch(cache, image_indexes)
    bundle.prepared = prepared


class SenseNovaDynamicFlow(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    def __init__(self, model_config, diffusion_model, shift: float = 3.0):
        self.patch_size = int(diffusion_model.patch_size)
        self.merge_size = int(1 / diffusion_model.downsample_ratio)
        self.base_noise_scale = float(diffusion_model.noise_scale)
        self.noise_scale_mode = str(diffusion_model.noise_scale_mode)
        self.noise_scale_base_image_seq_len = float(diffusion_model.noise_scale_base_image_seq_len)
        self.noise_scale_max_value = float(diffusion_model.noise_scale_max_value)
        super().__init__(model_config)
        self.set_parameters(shift=shift, multiplier=1000)

    def scale_for_shape(self, height: int, width: int) -> float:
        value = self.base_noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            grid_h, grid_w = height // self.patch_size, width // self.patch_size
            value *= math.sqrt(
                (grid_h * grid_w) / (self.merge_size**2) / self.noise_scale_base_image_seq_len
            )
            if self.noise_scale_mode == "dynamic_sqrt":
                value = math.sqrt(value)
        return min(value, self.noise_scale_max_value)

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = comfy.model_sampling.reshape_sigma(sigma, noise.ndim)
        scale = self.scale_for_shape(noise.shape[-2], noise.shape[-1])
        return sigma * (scale * noise) + (1.0 - sigma) * latent_image


class SenseNovaComfyModel(comfy.model_base.BaseModel):
    def __init__(self, handle: SenseNovaHandle):
        config = SenseNovaModelConfig(SenseNovaModelConfig.unet_config)
        super().__init__(config, model_type=comfy.model_base.ModelType.FLOW, device=torch.device(handle.device))
        self.diffusion_model = handle.model
        self.diffusion_model.config.t_eps = 0.02
        self.tokenizer = handle.tokenizer
        self.handle = handle
        self.model_sampling = SenseNovaDynamicFlow(config, self.diffusion_model, shift=3.0)
        self._active_bundles: dict[int, SenseNovaConditionBundle] = {}

    def extra_conds(self, **kwargs):
        spec = kwargs.get("sensenova_spec")
        if not isinstance(spec, SenseNovaBranchSpec):
            raise ValueError("SenseNova MODEL requires conditioning from SenseNova Conditioning.")
        return {"sensenova_condition": SenseNovaCondition(spec)}

    def _apply_model(self, x, sigma, sensenova_condition=None, **kwargs):
        if not isinstance(sensenova_condition, SenseNovaBranchSpec):
            raise TypeError("Missing SenseNova branch condition")
        bundle = sensenova_condition.bundle
        batch, _, height, width = x.shape
        shape = (batch, width, height)
        with bundle.lock:
            if bundle.prepared_shape != shape:
                bundle.clear()
                if bundle.images:
                    _prepare_edit_bundle(self, bundle, width, height, batch)
                else:
                    _prepare_text_bundle(self, bundle, width, height, batch)
                bundle.prepared_shape = shape
                self._active_bundles[id(bundle)] = bundle
        branch = bundle.prepared[sensenova_condition.branch]
        model = self.diffusion_model
        merge_size = int(1 / model.downsample_ratio)
        token_h = height // (model.patch_size * merge_size)
        token_w = width // (model.patch_size * merge_size)
        grid_h, grid_w = height // model.patch_size, width // model.patch_size
        device = x.device
        dtype = model.dtype
        x = x.to(dtype)
        # Upstream builds timesteps with torch.linspace, i.e. float32 even when
        # the model weights and image state are BF16.
        sigma_scalar = sigma.flatten()[0].to(device=device, dtype=torch.float32)
        native_t = 1.0 - sigma_scalar
        z = model.patchify(x, model.patch_size * merge_size)
        image_input = model.patchify(x, model.patch_size, channel_first=True)
        grid_hw = torch.tensor([[grid_h, grid_w]] * batch, device=device)
        image_embeds = model.extract_feature(
            image_input.view(batch * grid_h * grid_w, -1), gen_model=True, grid_hw=grid_hw
        ).view(batch, token_h * token_w, -1)
        t_expanded = native_t.expand(batch * token_h * token_w)
        timestep_embeddings = model.fm_modules["timestep_embedder"](t_expanded).view(
            batch, token_h * token_w, -1
        )
        if model.add_noise_scale_embedding:
            noise_scale = self.model_sampling.scale_for_shape(height, width)
            noise_tensor = torch.full_like(t_expanded, noise_scale / model.noise_scale_max_value)
            timestep_embeddings += model.fm_modules["noise_scale_embedder"](noise_tensor).view(
                batch, token_h * token_w, -1
            )
        image_embeds += timestep_embeddings
        v_tokens = model._t2i_predict_v(
            image_embeds,
            branch.indexes,
            {"full_attention": None},
            branch.cache,
            native_t,
            z,
            image_token_num=token_h * token_w,
            timestep_embeddings=timestep_embeddings,
            image_size=(width, height),
        )
        velocity = model.unpatchify(v_tokens, model.patch_size * merge_size, height, width)
        # Comfy CONST computes denoised = x - sigma * output; SenseNova uses
        # x_next = x + dt * velocity, hence the model output is -velocity.
        return self.model_sampling.calculate_denoised(sigma, -velocity.float(), x.float())

    def clear_conditioning_caches(self) -> None:
        for bundle in self._active_bundles.values():
            bundle.clear()
        self._active_bundles.clear()


class SenseNovaCleanupWrapper:
    def __init__(self, model: SenseNovaComfyModel):
        self.model = model

    def __call__(self, apply_model, args):
        return apply_model(args["input"], args["timestep"], **args["c"])

    def cleanup(self):
        self.model.clear_conditioning_caches()


def make_model_patcher(handle: SenseNovaHandle):
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    model = SenseNovaComfyModel(handle)
    model.device = offload_device
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )
    patcher.set_model_unet_function_wrapper(SenseNovaCleanupWrapper(model))
    return patcher


def make_pixel_vae():
    return comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)})


def patch_sampling(model, shift: float, cfg_norm: str, interval_start: float, interval_end: float):
    if not isinstance(model.model, SenseNovaComfyModel):
        raise TypeError("SenseNova Sampling Patch can only patch a SenseNova Loader MODEL.")
    patched = model.clone()
    original = patched.get_model_object("model_sampling")
    sampling = SenseNovaDynamicFlow(model.model.model_config, model.model.diffusion_model, shift=shift)
    sampling.set_noise_scale(original.noise_scale)
    patched.add_object_patch("model_sampling", sampling)

    def cfg_function(args):
        # cond/uncond are Comfy's noise predictions (x - denoised), which are
        # proportional to SenseNova velocity by the common sigma factor.
        cond = args["cond"]
        uncond = args["uncond"]
        scale_value = args["cond_scale"]
        sigma = float(args["sigma"].flatten()[0])
        native_t = 1.0 - sigma
        if native_t < interval_start or native_t > interval_end:
            return cond
        guided = uncond + scale_value * (cond - uncond)
        backend = model.model.diffusion_model
        patch = backend.patch_size * int(1 / backend.downsample_ratio)
        cond_tokens = backend.patchify(cond, patch)
        guided_tokens = backend.patchify(guided, patch)
        if cfg_norm == "global":
            factor = (
                torch.norm(cond_tokens, dim=(1, 2), keepdim=True)
                / (torch.norm(guided_tokens, dim=(1, 2), keepdim=True) + 1e-8)
            ).clamp(0, 1)
            guided_tokens *= factor
            guided = backend.unpatchify(guided_tokens, patch, cond.shape[-2], cond.shape[-1])
        elif cfg_norm == "channel":
            factor = (
                torch.norm(cond_tokens, dim=-1, keepdim=True)
                / (torch.norm(guided_tokens, dim=-1, keepdim=True) + 1e-8)
            ).clamp(0, 1)
            guided_tokens *= factor
            guided = backend.unpatchify(guided_tokens, patch, cond.shape[-2], cond.shape[-1])
        elif cfg_norm == "cfg_zero_star":
            from sensenova_u1.models.neo_unify.modeling_neo_chat import optimized_scale

            uncond_tokens = backend.patchify(uncond, patch)
            alpha = optimized_scale(
                cond_tokens.view(cond.shape[0], -1), uncond_tokens.view(uncond.shape[0], -1)
            )
            alpha = alpha.view(cond.shape[0], *([1] * (cond.ndim - 1))).to(cond.dtype)
            guided = uncond * alpha + scale_value * (cond - uncond * alpha)
            sample_sigmas = args["model_options"].get("transformer_options", {}).get("sample_sigmas")
            if sample_sigmas is not None and sigma >= float(sample_sigmas[0]) - 1e-6:
                guided = torch.zeros_like(guided)
        return guided

    patched.set_model_sampler_cfg_function(cfg_function, disable_cfg1_optimization=True)
    patched.model_options["sensenova_guidance"] = {
        "cfg_norm": cfg_norm,
        "interval_start": interval_start,
        "interval_end": interval_end,
    }
    return patched


class SenseNovaDualGuider(comfy.samplers.CFGGuider):
    def set_sensenova_conds(self, positive, middle, negative, text_cfg: float, image_cfg: float):
        self.inner_set_conds({"positive": positive, "middle": middle, "negative": negative})
        self.text_cfg = text_cfg
        self.image_cfg = image_cfg

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        out = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [self.conds["negative"], self.conds["middle"], self.conds["positive"]],
            x,
            timestep,
            model_options,
        )
        settings = model_options.get("sensenova_guidance", {})
        native_t = 1.0 - float(timestep.flatten()[0])
        if native_t < settings.get("interval_start", 0.0) or native_t > settings.get("interval_end", 1.0):
            return out[2]

        # Exact upstream edit rule: uncond + text*(positive-image) + image*(image-uncond).
        guided = out[0] + self.text_cfg * (out[2] - out[1]) + self.image_cfg * (out[1] - out[0])
        cfg_norm = settings.get("cfg_norm", "none")
        if cfg_norm in ("global", "channel") and (self.text_cfg > 1 or self.image_cfg > 1):
            backend = self.model_patcher.model.diffusion_model
            patch = backend.patch_size * int(1 / backend.downsample_ratio)
            positive_noise = backend.patchify(x - out[2], patch)
            guided_noise = backend.patchify(x - guided, patch)
            dims = (1, 2) if cfg_norm == "global" else -1
            factor = (
                torch.norm(positive_noise, dim=dims, keepdim=True)
                / (torch.norm(guided_noise, dim=dims, keepdim=True) + 1e-8)
            ).clamp(0, 1)
            guided_noise *= factor
            guided = x - backend.unpatchify(guided_noise, patch, x.shape[-2], x.shape[-1])
        return guided


def make_dual_guider(model, positive, middle, negative, text_cfg: float, image_cfg: float):
    guider = SenseNovaDualGuider(model)
    guider.set_sensenova_conds(positive, middle, negative, text_cfg, image_cfg)
    return guider


def checkpoint_assets_path(checkpoint: Path, metadata: dict[str, str] | None = None) -> Path:
    name = (metadata or {}).get("sensenova_assets_dir") or f"{checkpoint.stem}_assets"
    if Path(name).is_absolute() or len(Path(name).parts) != 1:
        raise ValueError(f"Invalid sensenova_assets_dir metadata: {name!r}")
    assets = checkpoint.parent / name
    if not (assets / "config.json").is_file():
        raise FileNotFoundError(
            f"SenseNova checkpoint assets not found: {assets}. Keep the *_assets directory beside the checkpoint."
        )
    return assets
