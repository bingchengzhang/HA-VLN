#!/usr/bin/env bash
# 在宿主机上用 vLLM 起一个 OpenAI 兼容的 Qwen2.5-VL 服务，和 Habitat 容器放在同一个 docker 网络里。
#
#   GPU=1 bash scripts/serve_vlm.sh                  # 起服务（GPU 换成师兄指定的卡）
#   LORA=$HOME/zbc/lora/nav bash scripts/serve_vlm.sh # 带微调后的 LoRA 适配器
#   docker logs -f zbc-vlm                            # 看加载进度
#   docker stop zbc-vlm && docker rm zbc-vlm          # 用完关掉，释放显存
#
# 服务只在 docker 网络 zbc-net 里可见，不对外开端口。Habitat 容器加入同一网络后用 http://zbc-vlm:8000/v1 访问：
#   docker network connect zbc-net zbc-havln
set -euo pipefail

WEIGHTS=${WEIGHTS:-$HOME/zbc/weights}
MODEL_DIR=${MODEL_DIR:-Qwen2.5-VL-7B-Instruct-AWQ}
GPU=${GPU:-0}
MEM=${MEM:-0.45}              # AWQ 版约 6 GB，0.45 够用并给 Habitat 留空间；bf16（微调合并后）约 16 GB，最好单独一张卡、MEM=0.85
IMAGE=${IMAGE:-vllm/vllm-openai:v0.8.5}

docker network inspect zbc-net >/dev/null 2>&1 || docker network create zbc-net

LORA_ARGS=()
MOUNT_LORA=()
if [ -n "${LORA:-}" ]; then
  MOUNT_LORA=(-v "$LORA:/lora:ro")
  LORA_ARGS=(--enable-lora --max-lora-rank 64 --lora-modules nav=/lora)
fi

docker run -d --name zbc-vlm --network zbc-net \
  --gpus "\"device=$GPU\"" --ipc=host \
  -v "$WEIGHTS:/weights:ro" "${MOUNT_LORA[@]}" \
  "$IMAGE" \
  --model "/weights/$MODEL_DIR" --served-model-name qwen-vl \
  --max-model-len 8192 --gpu-memory-utilization "$MEM" \
  --limit-mm-per-prompt image=4 "${LORA_ARGS[@]}"

echo "已启动 zbc-vlm。加载约需 1–3 分钟：docker logs -f zbc-vlm"
echo "带 LoRA 时，把 config 里 slow.model 改成 nav"
