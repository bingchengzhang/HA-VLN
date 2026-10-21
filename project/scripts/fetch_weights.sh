#!/usr/bin/env bash
# 下载权重到 $WEIGHTS（默认 ~/zbc/weights）。在宿主机上跑即可，之后挂进容器。
#
#   bash scripts/fetch_weights.sh            # YOLO-seg + Qwen2.5-VL-7B（AWQ 4-bit）
#   bash scripts/fetch_weights.sh --full-vlm # Qwen2.5-VL-7B 原版 bf16（约 16 GB 显存，单卡和 Habitat 共用会紧）
#   bash scripts/fetch_weights.sh --etpnav   # 另下 ETPNav 的深度编码器和路点预测器（需要全景，当前用不上）
#
# 只往 $WEIGHTS 里写，不改任何全局配置。需要代理时只在这条命令前加环境变量。
set -euo pipefail

WEIGHTS=${WEIGHTS:-$HOME/zbc/weights}
VLM=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
GET_ETPNAV=0
for a in "$@"; do
  case "$a" in
    --full-vlm) VLM=Qwen/Qwen2.5-VL-7B-Instruct ;;
    --etpnav) GET_ETPNAV=1 ;;
    *) echo "未知参数 $a"; exit 1 ;;
  esac
done
mkdir -p "$WEIGHTS"

# ---------- 1. 人体分割 ----------
YOLO_URL=https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-seg.pt
if [ ! -s "$WEIGHTS/yolov8m-seg.pt" ]; then
  echo "下载 YOLO-seg ..."
  curl -L --fail --retry 3 -o "$WEIGHTS/yolov8m-seg.pt.partial" "$YOLO_URL"
  mv "$WEIGHTS/yolov8m-seg.pt.partial" "$WEIGHTS/yolov8m-seg.pt"
fi

# ---------- 2. 慢回路视觉语言模型（ModelScope，国内直连） ----------
# modelscope 装在独立虚拟环境里，不碰系统 Python 和 conda base
if [ ! -x "$WEIGHTS/.venv/bin/modelscope" ]; then
  python3 -m venv "$WEIGHTS/.venv"
  "$WEIGHTS/.venv/bin/pip" install -q modelscope
fi
VLM_DIR="$WEIGHTS/$(basename "$VLM")"
if [ ! -f "$VLM_DIR/config.json" ]; then
  echo "下载 $VLM 到 $VLM_DIR ..."
  "$WEIGHTS/.venv/bin/modelscope" download --model "$VLM" --local_dir "$VLM_DIR"
fi

# ---------- 3. 可选：ETPNav 预训练权重 ----------
if [ "$GET_ETPNAV" = 1 ] && [ ! -d "$WEIGHTS/etpnav" ]; then
  echo "下载 ETPNav（深度编码器 ddppo-models、路点预测器 wp_pred）..."
  "$WEIGHTS/.venv/bin/modelscope" download --model admagic/ETPNav --local_dir "$WEIGHTS/etpnav"
fi

echo
echo "完成。目录："
du -sh "$WEIGHTS"/* 2>/dev/null | grep -v .venv || true
