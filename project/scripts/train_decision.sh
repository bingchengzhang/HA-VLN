#!/usr/bin/env bash
# 一键：微调 → 合并。在宿主机上跑（需要 setup_train_env.sh 建好的环境）。
#   GPU=1 bash scripts/train_decision.sh
# 两个 yaml 里写的是容器内的路径 /weights、/workspace；这里替换成宿主机上的 ~/zbc/weights、~/zbc/workspace。
set -euo pipefail
GPU=${GPU:-0}
VENV=${VENV:-$HOME/zbc/train-venv}
WEIGHTS=${WEIGHTS:-$HOME/zbc/weights}
WORKSPACE=${WORKSPACE:-$HOME/zbc/workspace}
cd "$(dirname "$0")/.."
source "$VENV/bin/activate"
TMP=$(mktemp -d)
for f in train_decision_lora export_merged; do
  sed -e "s#/weights#$WEIGHTS#g" -e "s#/workspace#$WORKSPACE#g" "configs/$f.yaml" > "$TMP/$f.yaml"
done
[ -f "$WORKSPACE/decision_data/merged/dataset_info.json" ] || { echo "没有训练数据：先跑 gen_decision_data.py 和 --merge"; exit 1; }
[ -f "$WEIGHTS/Qwen2.5-VL-7B-Instruct/config.json" ] || { echo "没有原版权重：bash scripts/fetch_weights.sh --full-vlm"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU llamafactory-cli train "$TMP/train_decision_lora.yaml"
CUDA_VISIBLE_DEVICES=$GPU llamafactory-cli export "$TMP/export_merged.yaml"
echo
echo "合并完成：$WEIGHTS/Qwen2.5-VL-7B-nav"
echo "换上新模型：docker stop zbc-vlm && docker rm zbc-vlm"
echo "            MODEL_DIR=Qwen2.5-VL-7B-nav MEM=0.85 GPU=<空闲的卡> bash scripts/serve_vlm.sh"
echo "离线评测：  python scripts/eval_decision_offline.py --data /workspace/decision_data/merged   （容器内）"
