# HA-VLN

RoboWorld 2026 Track 2（HA-VLN 2.0）分层视觉语言导航智能体。

实现、安装与运行方法见 [项目说明](project/README.md)，训练流程见 [训练文档](project/docs/TRAINING.md)。所有项目命令均从 `project/` 目录执行。

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
