#!/usr/bin/env bash
# 建微调用的独立环境（宿主机上，和 Habitat 容器完全分开，避免 torch/transformers 版本冲突）。
#   bash scripts/setup_train_env.sh            # 建在 ~/zbc/train-venv
# 只写 ~/zbc 下的目录，不碰系统 Python 和 conda base。
set -euo pipefail
VENV=${VENV:-$HOME/zbc/train-venv}
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install "llamafactory[torch,metrics]" bitsandbytes
"$VENV/bin/pip" freeze > "$VENV/requirements.lock"      # 记下实际版本，方便复现
"$VENV/bin/python" - <<'PY'
import torch, transformers, llamafactory
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "llamafactory", llamafactory.__version__)
PY
echo "激活：source $VENV/bin/activate"
