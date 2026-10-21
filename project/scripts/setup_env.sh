#!/usr/bin/env bash
# 在官方容器里（已跑过 /toolkit/scripts/setup_cma.sh）安装本项目的额外依赖。
# 原则：不升级、不替换镜像里的 torch / numpy / habitat，只补缺的包。
#
#   bash /workspace/havln_nav/scripts/setup_env.sh
set -euo pipefail

python - <<'PY'
import sys, habitat
assert habitat.__version__ == "0.1.7", habitat.__version__
assert sys.version_info >= (3, 8), f"ultralytics 需要 Python ≥ 3.8，镜像是 {sys.version}"
print("Python", sys.version.split()[0], "Habitat", habitat.__version__)
PY

# 把镜像现有的关键包版本钉死，防止 pip 顺手升级
CONSTRAINTS=$(mktemp)
pip freeze | grep -iE '^(torch|torchvision|numpy|opencv-python|opencv-python-headless|habitat.*)==' > "$CONSTRAINTS" || true
echo "钉住的版本："; cat "$CONSTRAINTS"

pip install -c "$CONSTRAINTS" "ultralytics==8.2.103" scipy pillow
pip check || echo "（pip check 有警告时先别慌，把输出发出来看）"

python - <<'PY'
import torch, ultralytics, scipy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__, "scipy", scipy.__version__)
PY
