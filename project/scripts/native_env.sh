# 每开一个服务器终端先执行：source <项目目录>/scripts/native_env.sh
# 只作用于当前终端，不写 .bashrc；缓存、临时文件都落在 $ZBC 里，不碰账号里的其他目录。
export ZBC=${ZBC:-/mnt/data/zbc}
export PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)     # 这个脚本所在的项目目录
export HPY=$ZBC/envs/havln/bin/python                 # Habitat 环境（Python 3.8，torch 2.0.1）
export TPY=$ZBC/envs/havln-train/bin/python           # 训练环境（Python 3.10，torch 2.6）
export TCLI=$ZBC/envs/havln-train/bin/llamafactory-cli
export DATA=$ZBC/data
export SRC=$ZBC/sources/cma-deps/HA-VLN
export IMG=$ZBC/image                                 # 从官方镜像拷出来的 /app 和 /opt/havln-challenge
export HAVLN_CHALLENGE_CONFIG=${HAVLN_CHALLENGE_CONFIG:-$IMG/app/config/challenge_submission.yaml}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$ZBC/cache/tmp PIP_CACHE_DIR=$ZBC/cache/pip PIP_CONFIG_FILE=/dev/null
export XDG_CACHE_HOME=$ZBC/cache/xdg XDG_CONFIG_HOME=$ZBC/config XDG_DATA_HOME=$ZBC/state XDG_STATE_HOME=$ZBC/state
export HF_HOME=$ZBC/cache/huggingface TORCH_HOME=$ZBC/cache/torch MPLCONFIGDIR=$ZBC/cache/matplotlib
export YOLO_CONFIG_DIR=$ZBC/config/ultralytics
export MAMBA_ROOT_PREFIX=$ZBC/tools/mamba CONDA_PKGS_DIRS=$ZBC/cache/conda CONDA_ENVS_PATH=$ZBC/envs CONDA_REGISTER_ENVS=false
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" \
         "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$ZBC/workspace" "$ZBC/weights" "$ZBC/logs"

# 官方镜像里的补丁版 habitat、habitat_extensions、评分包等：只对 Habitat 环境的 python 生效（HAVLN_PYTHONPATH）。
# 训练环境不需要它们，所以不直接改全局 PYTHONPATH；跑 Habitat 相关脚本时用 $HRUN 代替 $HPY。
if [ -d "$IMG" ] && [ -x "$HPY" ]; then
  export HAVLN_PYTHONPATH="$IMG/app:$("$HPY" "$PROJECT/scripts/native_paths.py" --root "$IMG")"
fi
export HRUN="env PYTHONPATH=${HAVLN_PYTHONPATH} $HPY"

cd "$PROJECT" || return
if [ "$GPU" = "none" ]; then
  unset GPU CUDA_VISIBLE_DEVICES          # 由 scripts/stage.py 自己挑卡
  echo "PROJECT=$PROJECT  GPU 由 stage.py 选择（允许 ${ALLOWED_GPUS:-5,6,7}）"
  return 0
fi
if [ -z "$GPU" ]; then
  read -r -p '师兄分给你的物理 GPU 编号: ' GPU
fi
if [[ "$GPU" =~ ^[0-9]+$ ]]; then
  export GPU CUDA_VISIBLE_DEVICES=$GPU     # 进程只看得到这一张卡，程序里的 --gpu 0 就是它
  echo "PROJECT=$PROJECT  GPU=$GPU"
  echo "Habitat 额外路径：${HAVLN_PYTHONPATH:-（未找到 $IMG，先完成第 3 步）}"
else
  echo "GPU 编号无效，没有设置 CUDA_VISIBLE_DEVICES"; unset GPU
fi
