# Comfy-Easy-SenseNova-U1

ComfyUI nodes for local SenseNova-U1 multimodal inference. See [README_CN.md](README_CN.md) for installation, node documentation, model storage layout, and precision/device guidance.

All nodes are registered under `eastmoe/Comfy-Easy-SenseNova-U1`. Model snapshots are stored in `ComfyUI/models/SenseNova/<subfolder>`.

SenseNova inference is isolated on the vendored Transformers `4.57.1` snapshot under `transformer_patch`; the plugin neither imports nor constrains ComfyUI's global Transformers installation.
