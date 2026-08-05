# SenseNova Transformers patch

本目录维护 SenseNova-U1 固定使用的 Hugging Face Transformers `4.57.1` 源码快照。

- `transformers_4571/` 来自 PyPI `transformers==4.57.1`。
- 包名改为 `transformers_4571`，避免覆盖 ComfyUI 与其他节点使用的全局 `transformers`。
- Auto 类的动态导入以及源码内的绝对导入均改为私有命名空间。
- 移除 wheel 面向全局安装环境的依赖版本门禁；缺失 API 仍会由实际导入明确报错。
- 上游许可证见 `LICENSE.transformers`。

更新快照时必须同步修改 `PINNED_VERSION`，并完成配置、tokenizer、权重加载及实际图像推理测试。
