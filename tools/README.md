# SenseNova checkpoint conversion

Convert an original BF16 Hugging Face snapshot without loading its tensors into RAM:

```bash
python tools/convert_hf_to_comfy_checkpoint.py \
  /path/to/SenseNova-U1-hf \
  /path/to/ComfyUI/models/checkpoints/SenseNova-U1.safetensors
```

The result is one `SenseNova-U1.safetensors` file. It contains the original
weights, the `vae.pixel_space_vae` sentinel used by native ComfyUI pixel
diffusion checkpoints, and a compressed tokenizer/config archive in
safetensors metadata. The converter preserves BF16 weights and does not
quantize them.

On first load, the plugin verifies the embedded archive and extracts it to
`ComfyUI/temp/SenseNova-U1_assets/`. It then creates a relative symbolic link
named `model.safetensors` pointing to the checkpoint. If symbolic links are not
permitted on Windows, it uses a same-volume hard link. The cache is rebuilt
when the checkpoint identity changes and can be deleted safely between loads.
If neither link type is available, loading fails instead of copying the full
weight file. Legacy sibling `_assets` directories are not supported; reconvert
old checkpoints with the current converter.

## Linear-only quantization

Run the quantizer with ComfyUI's Python environment. The source can be a Hugging
Face repo id, a local HF snapshot, or a converted safetensors checkpoint:

```bash
ComfyUI/venv/bin/python tools/quantize_checkpoint.py \
  sensenova/SenseNova-U1.5-8B-MoT \
  ComfyUI/models/checkpoints/SenseNova-U1.5-int8.safetensors \
  --method int8_convrot

ComfyUI/venv/bin/python tools/quantize_checkpoint.py \
  ComfyUI/models/checkpoints/SenseNova-U1.5-bf16.safetensors \
  ComfyUI/models/checkpoints/SenseNova-U1.5-mxfp8.safetensors \
  --method mxfp8
```

`--method` accepts `int8_convrot`, `mxfp8`, `w4a8_convrot`, and `mxfp4`.
Only weights belonging to actual `torch.nn.Linear` modules are quantized;
embeddings, norms, convolutions, biases, and other tensors stay in their source
precision. Use `--dry-run` to verify the source, private Transformers version,
and selected layers before processing a large checkpoint. `--include` and
repeatable `--exclude` accept regular expressions for controlled experiments.

The first three formats use the ComfyUI/comfy-kitchen quantization ABI. MXFP4
uses TorchAO and requires a PyTorch-compatible `torchao>=0.16`. Quantization is
streamed one source tensor at a time, but it still needs enough accelerator
memory for the largest individual Linear weight and temporary disk space for
the quantized payload. The output embeds the same compressed assets in its
single checkpoint file and is loaded by the same SenseNova Loader, which
continues to construct the model via the plugin's private patched Transformers
4.57.1.
