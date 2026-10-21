# HA-VLN

RoboWorld 2026 Track 2（HA-VLN 2.0）分层视觉语言导航智能体。

实现、安装与运行方法见 [项目说明](project/README.md)。所有项目命令均从 `project/` 目录执行。

## 架构与闭环

- [系统架构](project/docs/ARCHITECTURE.md)：感知、建图、候选规划、模型决策与动作执行。
- [训练与评估闭环](project/docs/CLOSED_LOOP.md)：专家采集、审核、微调、离线评估与真实仿真回测。
- [模型权重发布](project/docs/MODELS.md)：合并模型与 LoRA 的范围、发布方法及当前状态。
- [运行与训练说明](project/docs/TRAINING.md)：现有运行环境中的具体命令。

GitHub 保存代码、架构与闭环流程；训练权重放在 Hugging Face。环境文件、数据集、运行结果不上传。

已发布权重：[HA-VLN nav r0 LoRA](https://huggingface.co/bingchengzhang/HA-VLN-nav-r0-lora)，配合官方 Qwen2.5-VL-7B-Instruct 使用。[合并配置](project/configs/export_huggingface.yaml)可重建完整模型。

## 本地验证

```bash
cd project
python -m pip install numpy scipy pillow pytest
python -m pytest tests -q
```

这些测试不需要 Habitat；实际仿真、GPU 推理和训练仍需要项目说明中的运行环境。

## 仓库维护

- `main` 保存可追溯的代码版本；功能修改使用 `codex/<描述>` 或其他有意义的分支，通过 Pull Request 合并。
- Pull Request 自动执行测试；合并前检查测试结果与代码差异。
- 数据集、模型权重、训练输出、API 密钥和本机服务器访问脚本不上传。
- 外层运行记录、补丁和结果保留在本地；版本控制的实现位于 `project/`。
- 服务器通过拉取仓库更新代码；不要将服务器的临时修改作为唯一副本。
