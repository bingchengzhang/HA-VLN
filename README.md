# HA-VLN

RoboWorld 2026 Track 2 的分层视觉语言导航智能体（HA-VLN 2.0）。

几何模块负责建图、候选路线和安全约束。视觉语言模型只在决策点选择候选、判断子目标是否完成，以及是否停止。代码和流程在本仓库，训练权重在 Hugging Face。数据集、环境、运行结果和密钥不入库。

已发布权重：[HA-VLN nav r0 LoRA](https://huggingface.co/bingchengzhang/HA-VLN-nav-r0-lora)，配合 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)。完整模型用 [合并配置](project/configs/export_huggingface.yaml) 在本机导出。r0 在 20 条路线上的闭环成功率为 0，见 [闭环说明](project/docs/CLOSED_LOOP.md)。

## 文档

- [使用说明](project/README.md)：安装、运行、训练和评测。
- [系统架构](project/docs/ARCHITECTURE.md)
- [训练与评估闭环](project/docs/CLOSED_LOOP.md)
- [模型权重](project/docs/MODELS.md)
- [服务器训练操作](project/docs/TRAINING.md)

命令都在 `project/` 下执行。

## 本地测试

```bash
cd project
python -m pip install numpy scipy pillow pytest ruff
python -m ruff check . && python -m ruff format --check .
python -m pytest tests -q
```

这些测试不需要 Habitat。仿真、GPU 推理和训练需要使用说明里的运行环境。

## 维护

- `main` 保存可追溯的代码。功能修改走分支和 Pull Request。
- Pull Request 会跑上面的检查和测试。
- 不提交数据集、权重、训练输出、密钥，以及只在某台机器上使用的访问脚本。
- 服务器通过拉取仓库更新代码。
