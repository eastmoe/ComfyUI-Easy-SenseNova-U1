# SenseNova checkpoint conversion

Convert an original BF16 Hugging Face snapshot without loading its tensors into RAM:

```bash
python tools/convert_hf_to_comfy_checkpoint.py \
  /path/to/SenseNova-U1-hf \
  /path/to/ComfyUI/models/checkpoints/SenseNova-U1.safetensors
```

The result is `SenseNova-U1.safetensors` plus `SenseNova-U1_assets/`. The
checkpoint contains the original weights and `vae.pixel_space_vae`, the same
pixel-space VAE sentinel used by native ComfyUI pixel diffusion checkpoints.
The assets directory contains the local tokenizer/config and links back to the
checkpoint so the plugin's private patched Transformers 4.57.1 loader can load
the weights without converting the model implementation.

Keep both entries together. The safetensors file is the only weight payload;
tokenizer/config and the plugin's patched runtime are intentionally not encoded
as tensors. This converter preserves BF16 weights and does not quantize them.
