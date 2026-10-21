# 模型权重

## 发布范围

GitHub 保存实现与配置；Hugging Face 保存本项目训练后的模型。原版 Qwen 和 YOLO 从各自官方来源获取，不重复发布成自研模型。

| 产物 | 服务器目录 | 用途 |
|---|---|---|
| r0 合并模型 | `$ZBC/weights/Qwen2.5-VL-7B-nav-r0` | 完整推理权重、配置和 tokenizer，约 31 GiB |
| r0 LoRA | `$ZBC/workspace/lora/r0` | `adapter_model.safetensors` 约 154 MiB，需配合原版 Qwen 使用 |

基座：[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)。

## 下载与发布状态

- [r0 LoRA 权重](https://huggingface.co/bingchengzhang/HA-VLN-nav-r0-lora)：已上传，文件大小和 LFS SHA256 校验通过。
- r0 完整合并模型：上传处理中，尚未完成远端校验。

```bash
hf download bingchengzhang/HA-VLN-nav-r0-lora --local-dir /path/to/nav-r0-lora
```

LoRA 需要配合上面的原版 Qwen 基座使用，不能作为完整模型直接加载。仓库内的 `SHA256SUMS` 可用于下载后校验。

## 上传

在能读取上述权重目录的机器登录 Hugging Face：

```bash
python -m pip install -U huggingface_hub
hf auth login
python scripts/publish_weights.py --kind merged --folder "$ZBC/weights/Qwen2.5-VL-7B-nav-r0" --repo 用户名/HA-VLN-nav-r0 --dry-run
python scripts/publish_weights.py --kind merged --folder "$ZBC/weights/Qwen2.5-VL-7B-nav-r0" --repo 用户名/HA-VLN-nav-r0
python scripts/publish_weights.py --kind adapter --folder "$ZBC/workspace/lora/r0" --repo 用户名/HA-VLN-nav-r0-lora
```

上传脚本仅选择模型、tokenizer 与配置文件，不上传训练日志、数据集、优化器或环境目录。完成后核对远端文件大小及可用的 LFS SHA256，再更新本文的模型仓库链接。

Hugging Face 官方说明：[登录和 CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)、[上传文件](https://huggingface.co/docs/huggingface_hub/guides/upload)。

## 使用范围

模型用于 HA-VLN 决策研究。权重训练完成不代表闭环导航性能达标，详见 [闭环验证边界](CLOSED_LOOP.md)。使用时应遵循基座模型条款，以及训练数据对应的 [Matterport3D 使用协议](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf)；后者覆盖从数据训练得到的模型，限非商业学术用途。
