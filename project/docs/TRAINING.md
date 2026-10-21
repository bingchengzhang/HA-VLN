# 训练操作手册（师兄服务器，原生环境，不用 Docker）

本手册对应 `havln_nav` 当前版本。服务器是共享机器，规矩是：
- 所有写入都限定在 `/mnt/data/zbc`；
- 不用 sudo，不改 `.bashrc`、`.condarc`、`.gitconfig`，不往系统 Python 或 conda base 里装包；
- 不碰别人的进程和容器，只用师兄明确分给你的那张卡；
- 不用 Docker，你的账号也不在 docker 组里。

服务器连不上 Hugging Face 和 GitHub，所有外网下载都在 WSL 里完成，再用 scp 传上去。

**状态标注**：标 ✅ 的是你已经实际确认过的；标 ⚠️ 的是没在这台服务器上跑过的计划步骤，显存、耗时、兼容性都要现场确认。每一关没过就停下来，不要往下做。

## 总流程

```
⓪ 核实服务器现状 → ① 同步代码 → ② 终端变量
→ ③ 从官方镜像取出运行时文件 + check_env（当前卡在这一关）
→ ④ 训练环境 → ⑤ 数据与权重
→ ⑥ 不调模型的冒烟 rollout → ⑦ 原版模型闭环 + 专家诊断（各 20 条）
→ ⑧ 采集专家数据（先 20 条，再 2000 条）→ ⑨ 原版模型离线考试
→ ⑩ 20 步试训 → 正式训练 → ⑪ 合并 → ⑫ 微调后考试和闭环
→ ⑬ DAgger 第 1 轮（回到 ⑩）→ 提交
```

训练的是什么：只训练慢回路，也就是 Qwen2.5-VL-7B 在决策点上要做的那道选择题——选哪个候选、停不停、要不要转身看。建图、规划、执行都是程序，不参与训练。
标签由特权专家（`havln_agent/oracle.py`）根据训练集的参考路径自动生成，没有人工标注。
训练方法是监督微调（SFT）+ QLoRA：基座权重加载成 4-bit 后冻结，只训练插在语言模型线性层上的一组小矩阵（LoRA）；视觉编码器也冻结。

---

## ⓪ 核实服务器现状（服务器，只读不写）

```bash
ssh -p 6066 ffy@10.130.130.66
hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv     # 现在谁在用哪张卡
df -h /mnt/data/zbc; readlink -f /mnt/data/zbc
ls /mnt/data/zbc /mnt/data/zbc/sources /mnt/data/zbc/data /mnt/data/zbc/weights
find /mnt/data/zbc -maxdepth 3 -name havln_agent -type d                  # 已有项目代码在哪（可能不止一份）
/mnt/data/zbc/envs/havln/bin/python -c "import torch,habitat_sim;print(torch.__version__,habitat_sim.__version__)"
/mnt/data/zbc/envs/havln/bin/python -m pip download --no-deps -d /mnt/data/zbc/cache/tmp six -q && echo "服务器能用 pip"
```

这些命令全部只读，有几件事要记下来：
- **驱动版本**：训练环境用的是 cu118 版 torch，驱动需要 ≥ 520。4090 的驱动一般都满足，但还是要看一眼。
- **哪张卡空、哪张卡是你的**：卡空着不代表你能用，以师兄明确分给你的编号为准。
- **项目代码有几份**：只该有一份，有多份就先确认哪份是你在用的。
- **服务器能不能用 pip**：决定第 ④ 步是在服务器上直接装，还是在 WSL 里打包好再传上来。
- **剩余磁盘**：整个流程大约需要 80 GB，见第 ⑤ 步末尾的预算。

`find` 找不到 `havln_agent`，说明服务器上还没有项目代码，第 ① 步直接放到 `sources/havln_nav` 就行。

## ① 同步代码

> ⚠️ 你 WSL 里同步的是**上一版** zip（117,098 字节，SHA256 `8544071a…`），不是这一版。这一版新增了 `docs/TRAINING.md`、`scripts/native_env.sh`、`scripts/native_paths.py`、`scripts/extract_image_files.py`、`scripts/make_train_configs.py`，还改了数据划分和 prompt 对齐。先把桌面上的 zip 换成新版再同步。
> Codex 的 `prepare_decision_training.py` 不要用：它写出的验证文件格式和 `eval_decision_offline.py` 读的不一样，第 ⑨ 步会直接报错。它的功能现在由 `gen_decision_data.py --merge --reference` 和 `make_train_configs.py` 实现。`sync_server_code.py` 同步代码可以继续用。

不用同步工具的话，**WSL** 里执行：

```bash
scp -P 6066 /mnt/c/Users/lenovo/Desktop/havln_nav.zip ffy@10.130.130.66:/mnt/data/zbc/installers/
```

然后在**服务器**上解压。代码已经存在时，先备份再覆盖（第 ⓪ 步找到的位置如果不是 `sources/havln_nav`，就把下面的路径换掉）：

```bash
cd /mnt/data/zbc/sources
[ -d havln_nav ] && mv havln_nav "/mnt/data/zbc/backups/havln_nav-$(date +%Y%m%d-%H%M)"
python3 -c "import zipfile; zipfile.ZipFile('/mnt/data/zbc/installers/havln_nav.zip').extractall('.')"
ls havln_nav/docs/TRAINING.md && echo "新版已就位"
```

zip 的顶层目录就是 `havln_nav`。旧版移到 `backups/` 里，里面要是有你改过的东西，可以去那里找回来。之后改用 git 仓库时，这一步换成 `git pull`。

## ② 每个新终端先执行

```bash
source /mnt/data/zbc/sources/havln_nav/scripts/native_env.sh
```

这个脚本做了几件事：
- 设置路径变量：`$ZBC`、`$PROJECT`（项目目录，根据脚本自己的位置推出来）、`$HPY`（Habitat 环境的 python）、`$TPY`（训练环境的 python）、`$TCLI`（llamafactory-cli）、`$DATA`、`$SRC`、`$IMG`（从镜像取出的文件放在这里）。
- 把 pip、HF、torch、YOLO 的缓存和临时目录都指到 `$ZBC` 里面，不往账号的 home 目录写东西。
- 设置 **`$HRUN`**：等于 `$HPY` 再加上镜像运行时的包路径（补丁版 habitat、habitat_extensions、评分包）。**所有和 Habitat 有关的脚本都用 `$HRUN` 来跑。** 这些包路径只对这一条命令生效，不会影响训练环境。
- 问你 GPU 编号，然后设置 `CUDA_VISIBLE_DEVICES`。设置之后，进程里只看得到这一张卡，后面命令里写的 `--gpu 0` 和 `--gpu-ids 0` 指的就是它，**不是物理上的 0 号卡**。

这些设置只在当前终端有效。每开一个新终端都要重新 source，否则会用错 python，或者跑到别人的卡上。

## ③ 从官方镜像取出运行时文件 + 环境验收（必须先过这一关）

**为什么要做这一步**：官方的 `export_cma_submission.py` 是在镜像里跑的，它依赖镜像里的三样东西：
- `/app/config/challenge_submission.yaml`：比赛环境配置，我们的适配器也是照着它建环境的；
- `/app/habitat-lab`：打过补丁的 habitat。官方脚本会专门校验 `habitat_simulator.py` 的哈希，说明这个文件被改过，行人很可能就是在这里接进去的；
- `/opt/havln-challenge/scoring`：评分包 `havln_scoring`，包括六动作任务配置 `six_action_task.yaml`、提交校验器 `havln-validate`，以及 phase1 的参考清单。

你现在的 `envs/havln` 装的是原版 habitat-lab 0.1.7，没有这些补丁。之前导入成功，是因为改了 `sys.path` 绕过去的；那只能说明类能 import，不能说明环境能正常建起来、行人能正常加载。

镜像是公开的：`ghcr.io/jostarxiong/havln-challenge-2026@sha256:e1a0544f…`（官方 README 里写的就是这个摘要）。
**不需要 Docker**：`extract_image_files.py` 只用 Python 标准库，按镜像仓库协议匿名下载每一层，从中解出需要的目录。

**WSL** 里执行（⚠️ 这个脚本只用合成数据测过解层逻辑，还没连真实仓库跑过；整个镜像可能有十几 GB，下载需要时间和磁盘）：

```bash
cd /home/bccc/workspace/havln
python3 havln_nav/scripts/extract_image_files.py --out image-files --cache image-blobs
ls image-files/app image-files/opt/havln-challenge
grep -E "habitat_extensions/__init__|vlnce_baselines/__init__|havln_scoring/__init__|bin/havln-validate|habitat_simulator.py$" image-files/index.txt
```

- `--cache` 用来保存下载下来的层文件，每层下完会校验 sha256。中途断了就重跑同一条命令，已经下好的层会跳过。
- 默认取出 `/app`、`/opt/havln-challenge`，以及镜像 site-packages 里所有的 `.pth` 文件（`.pth` 记录了包装在哪里）。`index.txt` 是镜像里全部文件的路径清单。
- `grep` 那一行用来确认几个关键包和校验器在镜像里的位置。如果 `habitat_extensions` 或 `havln-validate` 的路径不在 `/app` 和 `/opt/havln-challenge` 底下（比如装在镜像 python 的 site-packages 里），就用 `--prefix` 把那个目录也取出来：`--prefix app/ opt/havln-challenge/ <那个目录>/`。注意要换一个新的 `--out` 目录。
- `image_config.json` 里的 `config.Env` 记录了镜像预设的环境变量（PYTHONPATH 等），对照着看包是怎么接进去的。

查看配置里引用了哪些镜像内路径，然后打包上传：

```bash
grep -rnE "/app|/opt/havln-challenge|/data/havln2|/workspace|/weights" image-files/app/config | head -50
tar -czf image-files.tar.gz -C image-files .
scp -P 6066 image-files.tar.gz ffy@10.130.130.66:/mnt/data/zbc/installers/
```

**服务器**（已 source native_env.sh）：

```bash
[ -e "$IMG" ] && echo "$IMG 已存在，先确认内容" || { mkdir -p "$IMG" && tar -xzf "$ZBC/installers/image-files.tar.gz" -C "$IMG"; }
# 配置里的镜像内绝对路径改成服务器上的位置（只改拷出来的副本）
grep -rlE "/app/|/opt/havln-challenge/|/data/havln2" "$IMG/app/config" "$IMG/opt/havln-challenge" --include="*.yaml" \
  | xargs -r sed -i -e "s#/app/#$IMG/app/#g" -e "s#/opt/havln-challenge/#$IMG/opt/havln-challenge/#g" -e "s#/data/havln2#$DATA#g"
grep -rnE "(^|[^a-z])/app/|/opt/havln-challenge|/data/havln2" "$IMG/app/config" --include="*.yaml" | grep -v "$IMG" || echo "配置里已无镜像内路径"
source "$PROJECT/scripts/native_env.sh"           # 重新 source，让它找到 $IMG 里的包
$HRUN scripts/native_paths.py --root "$IMG" --explain
```

- `sed` 只改 yaml 配置，不碰 `.py` 源码。如果源码里也写死了 `/app` 这类路径，check_env 时会报找不到文件，到时候再按报错逐个处理，不要一开始就批量改代码。
- `native_paths.py --explain` 会列出每个关键包在 `$IMG` 里找到的位置。`habitat`、`habitat_extensions`、`vlnce_baselines`、`havln_scoring` 都应该找得到。`HASimulator` 找不到不要紧，适配器会从 `$SRC` 里找。

补丁到底改了什么，可以拿镜像版和你装的原版 habitat 比一下（只读）：

```bash
diff -rq "$ZBC/sources/habitat-lab-0.1.7/habitat" "$(dirname "$($HRUN -c 'import habitat;print(habitat.__file__)')")" | head -30
$HRUN -c "import habitat,habitat_extensions,vlnce_baselines;print(habitat.__file__)"
```

第二行打印出的路径应该在 `$IMG/app/habitat-lab` 下面，这说明 `$HRUN` 用的确实是补丁版 habitat。

装智能体额外需要的依赖（YOLO 分割、scipy），同时锁住 Habitat 环境里已有的 torch 和 numpy，防止 pip 顺手升级它们：

```bash
"$HPY" -m pip freeze > "$ZBC/workspace/havln-before-agent.lock"          # 出问题时对照这个回滚
"$HPY" -m pip freeze | grep -iE "^(torch|torchvision|numpy|opencv-python[a-z-]*)==" > "$ZBC/workspace/havln-constraints.txt"
"$HPY" -m pip install -c "$ZBC/workspace/havln-constraints.txt" ultralytics==8.2.103 scipy==1.10.1 Pillow==10.4.0
"$HPY" -m pip check
```

`-c` 是约束文件，版本对不上时 pip 会报冲突并停下，而不是偷偷换版本。报了冲突就停下来检查，不要去掉约束硬装。
服务器不能用 pip 的话，在 WSL 里用 **Linux 的 Python 3.8** 执行 `pip download ultralytics==8.2.103 scipy==1.10.1 Pillow==10.4.0 -d wheels --python-version 3.8 --platform manylinux2014_x86_64 --only-binary=:all:`，把 wheels 目录传上去，再到服务器上执行 `pip install --no-index --find-links wheels -c ...`。

**环境验收**：

```bash
$HRUN scripts/check_env.py --split val_unseen --data-root "$DATA" --source-root "$SRC" --gpu 0 2>&1 | tee "$ZBC/logs/check_env.log"
```

这一步会真的建环境、reset、走几步，然后打印传感器参数（RGB/深度的分辨率、HFOV、深度范围、相机高度和俯仰）、动作集、navmesh 的最大台阶和坡度，以及多楼层场景所占的比例。

**通过标准**：
- reset 和 step 都成功；
- 动作集正好是 6 个；
- 深度不是全零；
- 位姿随动作变化（前进 0.25 m、转 15°）；
- 日志里能看到行人加载。

通过之后，把打印出来的参数和 `havln_agent/config.py` 里标了 VERIFY 的值逐项对一遍，不一致就改 config。

**常见失败**：
- **找不到 `challenge_submission.yaml`**：`$IMG` 解压的位置不对，或者 `HAVLN_CHALLENGE_CONFIG` 没设对。
- **`ModuleNotFoundError: havln_scoring` / `habitat_extensions`**：`native_paths.py --explain` 有哪一项没找到，回到 WSL 用 `--prefix` 把缺的目录补取出来。
- **报某个 yaml 或数据文件找不到，路径以 `/app`、`/opt`、`/data/havln2` 开头**：漏改的路径，按上面的 sed 规则再改一次。
- **habitat_sim 报 C++ 层的错**：镜像里的 habitat-sim 可能也和你装的不一样。把 `index.txt` 里 `habitat_sim` 所在的目录取出来比对。这种情况现场排查，不要自己重编译。
- **`$DATA/recompute_navmesh` 不存在**：先看报错是不是因为它。这个目录在官方流程里怎么生成的，我这边没法确认，需要看 HA-VLN 仓库的数据说明，或者问主办方。

## ④ 训练环境（和 Habitat 分开） ⚠️

Habitat 环境是 Python 3.8 + torch 2.0.1，装不了 LLaMA-Factory 0.9.3，所以训练另外建一个 Python 3.10 的环境。
两个环境之间只通过文件交换数据，通过 HTTP 调用模型，互不影响。

服务器能用 pip 时，直接在**服务器**上装：

```bash
"$ZBC/tools/bin/micromamba" --no-rc create -p "$ZBC/envs/havln-train" -c conda-forge python=3.10 pip -y
"$TPY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
"$TPY" -m pip install llamafactory==0.9.3 transformers==4.51.3 peft==0.15.2 accelerate==1.7.0 \
    datasets==3.6.0 trl==0.9.6 bitsandbytes==0.45.5 numpy==1.26.4 huggingface-hub==0.30.2
"$TPY" -m pip check
"$TPY" -c "import torch,bitsandbytes;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0),'bf16',torch.cuda.is_bf16_supported())"
"$TPY" -m pip freeze > "$ZBC/workspace/train-requirements.lock"
```

- **版本为什么全部钉死**：LLaMA-Factory 0.9.3 对 transformers、peft、trl、accelerate、datasets 都设了版本上限，上面这组都落在它的范围内。换成最新版，容易碰到参数名改掉的问题。
- **最后一行检查**：应该输出 `True`、`NVIDIA GeForce RTX 4090`、`bf16 True`。
- **`pip freeze`**：存一份锁文件，以后重建环境、或者队友复现时照着装。

服务器不能用 pip 时，在 **WSL**（Linux x86_64，**不是** Windows）里建同样版本的环境，打包后上传：

```bash
cd /home/bccc/workspace/havln
MAMBA_ROOT_PREFIX=$PWD/mamba-pack ./downloads/micromamba-linux-64 --no-rc create -p $PWD/train-pack-env -c conda-forge python=3.10 pip -y
E=$PWD/train-pack-env/bin/python
$E -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
$E -m pip install llamafactory==0.9.3 transformers==4.51.3 peft==0.15.2 accelerate==1.7.0 \
    datasets==3.6.0 trl==0.9.6 bitsandbytes==0.45.5 numpy==1.26.4 huggingface-hub==0.30.2 conda-pack
$E -m pip check && $PWD/train-pack-env/bin/conda-pack -p $PWD/train-pack-env -o havln-train-linux.tar.gz
scp -P 6066 havln-train-linux.tar.gz ffy@10.130.130.66:/mnt/data/zbc/installers/
```

然后在**服务器**上解压：

```bash
[ -e "$ZBC/envs/havln-train" ] && echo "已存在，先确认" || { mkdir -p "$ZBC/envs/havln-train" \
  && tar -xzf "$ZBC/installers/havln-train-linux.tar.gz" -C "$ZBC/envs/havln-train" && "$ZBC/envs/havln-train/bin/conda-unpack"; }
```

解压完再跑一遍上面的 `pip check` 和 GPU 检查。

## ⑤ 数据与权重（在 WSL 下载，服务器连不上 HF） ⚠️

**WSL**：

```bash
cd /home/bccc/workspace/havln/downloads
# 训练集：版本钉在官方工具包用的同一个 HF 提交（官方下载清单只包含 val）
curl -fL --retry 3 -o train.json.gz \
  https://huggingface.co/datasets/fly1113/HA-VLN/resolve/c5f1cec167964bda3ba343acc3a18e7aef8540dd/HA-R2R/train/train.json.gz
echo "568f966edab1fde96891b8c831bf5d1ba83f3658c4d868c8809cc779805cbf19  train.json.gz" | sha256sum -c -

# 原版 bf16 Qwen2.5-VL-7B-Instruct（约 16 GB；训练和合并都要用它，AWQ 量化版不能当训练基座）
python3 -m venv hfdl && hfdl/bin/pip install -q huggingface-hub==0.30.2
hfdl/bin/python - <<'PY'
from huggingface_hub import snapshot_download, HfApi
repo = "Qwen/Qwen2.5-VL-7B-Instruct"
rev = HfApi().model_info(repo).sha
d = snapshot_download(repo_id=repo, revision=rev, local_dir="Qwen2.5-VL-7B-Instruct")
open(d + "/DOWNLOAD_REVISION.txt", "w").write(rev + "\n")
print(d, rev)
PY

# YOLO 分割权重（快回路用来识别行人）
curl -fL --retry 3 -o yolov8m-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-seg.pt

scp -P 6066 train.json.gz ffy@10.130.130.66:/mnt/data/zbc/installers/
scp -P 6066 -r Qwen2.5-VL-7B-Instruct yolov8m-seg.pt ffy@10.130.130.66:/mnt/data/zbc/weights/
```

`DOWNLOAD_REVISION.txt` 记下了这次下载对应的 HF 提交号，以后复现结果时要用。

**服务器**：

```bash
mkdir -p "$DATA/HA-R2R/train" && cp "$ZBC/installers/train.json.gz" "$DATA/HA-R2R/train/"
echo "568f966edab1fde96891b8c831bf5d1ba83f3658c4d868c8809cc779805cbf19  $DATA/HA-R2R/train/train.json.gz" | sha256sum -c -
"$TPY" - <<'PY'
import json, gzip
from pathlib import Path
d = Path("/mnt/data/zbc/weights/Qwen2.5-VL-7B-Instruct")
idx = json.loads((d / "model.safetensors.index.json").read_text())
miss = [n for n in set(idx["weight_map"].values()) if not (d / n).is_file()]
print("模型分片缺失:", miss or "无")
print("train episodes:", len(json.load(gzip.open("/mnt/data/zbc/data/HA-R2R/train/train.json.gz", "rt"))["episodes"]))
PY
ls "$DATA/scene_datasets/mp3d" | head -3; ls "$DATA/scene_datasets/mp3d" | wc -l     # MP3D 场景（Habitat 子集 90 个）
ls "$DATA/HAPS2_0" | head -3; ls "$DATA/Multi-Human-Annotations/human_motion.json" "$DATA/HA-R2R-tools"
```

比赛数据之前已经在 WSL 下好、校验过了。服务器上要确认的是：已经上传、已经解压，而且解压后的布局和下面一致：

```
$DATA/HA-R2R/{train,val_seen,val_unseen}/   $DATA/HA-R2R-tools/          $DATA/Multi-Human-Annotations/
$DATA/HAPS2_0/   $DATA/scene_datasets/mp3d/<scan>/<scan>.glb   $DATA/checkpoints/   $DATA/recompute_navmesh/（见第 ③ 步）
```

缺哪项就从 WSL 的 `downloads/competition-data/` 传哪项，传完先用 `TRANSFER_SHA256SUMS` 校验，再解压。

磁盘预算（大约）：

| 项目 | 大小 |
|---|---|
| 原版模型 | 16 GB |
| 每轮合并后的模型 | 16 GB |
| 训练环境 | 10 GB |
| 镜像取出的文件 | 几 GB |
| 2000 条 episode 的采集图片 | 几 GB |
| checkpoint | 每个几百 MB |

## ⑥ 冒烟：先不调模型跑通 20 条

```bash
$HRUN scripts/run_rollout.py --data-root "$DATA" --source-root "$SRC" \
  --splits val_unseen --limit 20 --gpu-ids 0 --no-llm --local-metrics \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --output-dir "$ZBC/workspace/runs/smoke-nollm"
```

`--no-llm` 表示决策点不调大模型，改用兜底规则（选最接近直行、没走过的候选）。这一步只验证"仿真 + 建图 + 规划 + 执行 + 导出动作 + 本地打分"整条链路能跑通，分数没有参考意义。
看 `worker-gpu0.log` 有没有报错，`logs/` 里每条 episode 的事件日志是否合理（有没有原地打转、撞墙），以及最后打印的碰撞率。
这一步不需要训练环境，也不需要模型服务，所以放在最前面做。

## ⑦ 原版模型闭环 + 专家诊断

先生成本轮的全部配置：

```bash
"$TPY" scripts/make_train_configs.py --run r0 --root "$ZBC" --dataset "$ZBC/workspace/decision_data/r0-split"
```

```bash
export CFG=$ZBC/workspace/train-configs/r0
```

这时 `r0-split` 里还没有数据，脚本会给出提示，这是正常的。服务和智能体的配置现在就能用；训练配置里只记了数据目录的路径，等 ⑧ 划分完数据，内容自然就对上了，不用重新生成。
`export CFG=...` 每个新终端都要执行一次。

**卡的分工**：模型服务固定放 **6 号卡**（终端 A），仿真、采集都放 **4 号卡**（终端 B）。两边互不抢显存，可以同时跑。

**终端 A**（专门跑模型服务，不要在这个终端里敲别的命令）：

```bash
GPU=6 source /mnt/data/zbc/sources/havln_nav/scripts/native_env.sh
API_HOST=127.0.0.1 API_PORT=18080 API_MODEL_NAME=zbc-r0-base "$TCLI" api "$ZBC/workspace/train-configs/r0/api-base.yaml"
```

这里用 LLaMA-Factory 自带的 HF 后端起一个 OpenAI 兼容的接口，原来的 `serve_vlm.sh` 要用 Docker，这台服务器用不了。
HF 后端比 vLLM 慢得多，每次决策可能要几秒，所以这里测到的耗时不代表最终部署的速度。
基座和后面的微调模型都用 4-bit 推理，这样比较时两边的精度条件一致。
服务只绑定 `127.0.0.1`，外面访问不到。如果 18080 端口被占用，就换一个端口，同时把 `agent-*.json` 里的 `base_url` 改成同一个端口。不要停掉别人的服务。

**终端 B**：

```bash
GPU=4 source /mnt/data/zbc/sources/havln_nav/scripts/native_env.sh; export CFG=$ZBC/workspace/train-configs/r0
curl --noproxy '*' -sf http://127.0.0.1:18080/v1/models && echo "服务正常"

# 把这 20 条 val 指令拆成子目标（只拆要用到的，HF 后端很慢，没必要全量拆）
$HRUN scripts/compile_instructions.py --data-root "$DATA" --splits val_unseen --limit 20 --workers 1 \
  --cfg "$CFG/agent-base.json" --out "$ZBC/workspace/plans/val.jsonl"

# 原版模型做决策
$HRUN scripts/run_rollout.py --data-root "$DATA" --source-root "$SRC" --splits val_unseen --limit 20 --gpu-ids 0 \
  --cfg "$CFG/agent-base.json" --plans "$ZBC/workspace/plans/val.jsonl" \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --local-metrics --output-dir "$ZBC/workspace/runs/r0-base-20"

# 专家代替模型做决策（诊断用，禁止提交）
$HRUN scripts/run_rollout.py --data-root "$DATA" --source-root "$SRC" --splits val_unseen --limit 20 --gpu-ids 0 \
  --no-llm --oracle-decisions --plans "$ZBC/workspace/plans/val.jsonl" \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --output-dir "$ZBC/workspace/runs/oracle-20"
```

- `--noproxy '*'`：服务器上如果设置了代理环境变量，curl 访问本机服务也会被转发到代理，所以这里显式绕开。智能体的客户端默认就不走代理。
- 拆指令的输出里如果 `fallback` 很多，说明模型服务有问题，比如输出不是 JSON、请求超时。要先查清原因，不要把兜底结果当成拆解成功。
- 两次 rollout 对比着看：
  - **专家跑得好、原版差**：说明问题在慢回路的判断上，训练正好就是解决这个的，继续往下做。
  - **专家也差**：说明建图、候选生成或执行有问题，训练救不了。先看汇总里的 `candidate_coverage`（候选里有没有正确方向）、`wrong_floor_rate`、碰撞率，再翻 `logs/` 里失败的 episode。
- 20 条只是粗筛；专家本身也是启发式规则，它的分数不是严格的上限。

服务可以继续开着，⑧ 拆指令还要用它。以后要停服务时，就在终端 A 按 **Ctrl-C**；不要用 `pkill python`，那样会连别人的进程一起杀掉。

## ⑧ 采集专家数据

先拆好要采集的那批训练 episode 的指令。训练数据里的 prompt 必须带上和推理时一样的子目标清单，否则模型训练时看到的输入和实际使用时不一样。

```bash
# 列出 seed=0、limit=2000 时会抽到的 episode（和正式采集用的是同一个抽样函数）
$HRUN scripts/gen_decision_data.py --data-root "$DATA" --limit 2000 --seed 0 --list-episodes \
  --out "$ZBC/workspace/plans/train-r0-ids.txt"
```

拆指令需要模型服务（终端 A 的 api-base 应该还开着，停了的话就重新起）。在终端 B 执行：

```bash
$HRUN scripts/compile_instructions.py --data-root "$DATA" --splits train --workers 1 \
  --episode-list "$ZBC/workspace/plans/train-r0-ids.txt" \
  --cfg "$CFG/agent-base.json" --out "$ZBC/workspace/plans/train.jsonl"
```

这一步很长：2000 条指令，每条都要调用一次模型。中途可以中断，重跑时会自动跳过已经拆好的。后面的采集不需要模型，服务开着也不碍事，它在 6 号卡上，采集用的是 4 号卡。

先小规模试采：

```bash
$HRUN scripts/gen_decision_data.py --data-root "$DATA" --source-root "$SRC" \
  --plans "$ZBC/workspace/plans/train.jsonl" --cfg "$CFG/agent-base.json" \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --gpu 0 \
  --limit 20 --perturb 0.15 --seed 0 --out "$ZBC/workspace/decision_data/smoke"
```

- 在训练集上走 20 条路线。每到一个决策点，专家给出正确答案，脚本把"模型会看到的输入（文字 + 图片）"和"专家答案"一起存成一道题。
- `--perturb 0.15`：在有多个可走候选的决策点，以 15% 的概率故意执行一个错误的候选，但标签仍然记专家的答案。这样数据里会包含"走偏了之后怎么回来"的状态。停止和观察这两类决策点不做扰动。
- 不加 `--dagger` 时完全不调用模型，所以不需要服务。

**人工抽查**：打开 `smoke/shards/*.jsonl` 和 `smoke/images/`，抽 30 道题来看，重点看跨楼层、走偏后回正、停止这几类。
检查专家的答案是否合理，以及停止题的 `goal_box` 是否框在终点附近。
专家的框是几何投影算出来的近似位置，不是真实的物体标注，被遮挡时也会照样画出来。标签错了，训练再多也修不好，所以这一步不能省。

确认没问题后正式采集（换一个新目录，不要和试采的混在一起）：

```bash
$HRUN scripts/gen_decision_data.py --data-root "$DATA" --source-root "$SRC" \
  --plans "$ZBC/workspace/plans/train.jsonl" --cfg "$CFG/agent-base.json" \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --gpu 0 \
  --limit 2000 --perturb 0.15 --seed 0 --out "$ZBC/workspace/decision_data/r0"

$HRUN scripts/gen_decision_data.py --merge "$ZBC/workspace/decision_data/r0" \
  --out "$ZBC/workspace/decision_data/r0-split"
```

- 2000 是 episode 数，不是样本数。每条 episode 大约产生 5–15 个决策点，实际样本数看划分时打印的数字。
- 采集按场景分片保存，中断后用同样的命令重跑，会跳过已完成的分片。如果改了代码、参数或 seed，要换一个新的输出目录，不要和旧分片混在一起。
- `--merge` 的作用：
  - 按**场景**留出约 5% 作为验证集，同一场景的题不会同时出现在训练集和验证集里，这样验证集考的是没见过的房子。
  - 检查每道题的图片都存在，并且 `<image>` 占位符的个数和图片张数一致，对不上就报错退出。
  - 写出 LLaMA-Factory 需要的 `dataset_info.json`，以及离线评测用的 `decision_val_raw.json`、`heldout_scans.json`（验证场景清单）、`manifest.json`（来源和标签分布）。
- 划分后的 json 用绝对路径引用 `r0/images` 里的图片，所以 `r0` 目录不能删。
- 看一下 `manifest.json` 里的标签分布。stop 标签通常远少于 choose（每条 episode 只有一次停止）；如果 stop 少到只有个位数，就多采一些。

## ⑨ 训练前先考原版

终端 A 的 api-base 要开着；在终端 B 执行：

```bash
$HRUN scripts/eval_decision_offline.py --data "$ZBC/workspace/decision_data/r0-split" \
  --cfg "$CFG/agent-base.json" --limit 1000000
```

这一步把验证集的每道题发给模型，对比专家答案，不需要仿真器，几百道题几分钟到几十分钟就能跑完。`--limit 1000000` 的意思是考全部题目（默认只考 300 题）。
输出的指标：
- `json_ok`：输出能解析、格式合法的比例。原版模型这一项可能就不高，训练首先要把它拉到接近 1。
- `choose_acc`：专家要选候选的题里，模型选对编号的比例。
- `stop_recall`：该停的时候停了的比例；`stop_false`：不该停的时候停了的比例。后者每高一点，都会直接造成失败。
- `observe_match`：专家要求转身看的时候，模型也要求看的比例（只看有没有要求，不看方向对不对）。
- `box_center_err_px`：停止题里模型画的框和专家的框，中心点相差多少像素（在 336×336 的图上）。

结果会存成 `offline_eval_zbc-r0-base.json`。这是训练前的基线，后面每一轮都用同一份验证题来比。
考完回终端 A 按 Ctrl-C 停掉服务。下面的训练要同时用 4、6 两张卡，所以服务必须先停。

## ⑩ 训练

先试训 20 步：

```bash
CUDA_VISIBLE_DEVICES=4 "$TCLI" train "$CFG/smoke.yaml"
```

最多取 64 条样本，每步更新一次，共 20 步。用来检查数据格式、图片读取、loss 是否正常、反向传播是否能跑、checkpoint 能不能保存。输出在 `lora/r0-smoke`，和正式训练的输出分开放。
试训只用 4 号一张卡，这样排查问题更简单。另开一个终端，运行 `nvidia-smi -i 4,6 -l 2` 看显存。
- **loss 是 nan 或 0**：通常是答案被截断了，或者模板选错了。检查 `cutoff_len`，以及日志里打印的样本示例。
- **OOM（显存不够）**：先确认终端 A 的服务已经停掉。真的不够时，减少每道题的图片数（`build_images` 的 `max_images`），而且推理时也要同步改成一样的设置。不要只在训练这边单独缩小 `image_max_pixels`：图片尺寸和 `goal_box` 的坐标系是绑定的。梯度累积不能降低单条样本占用的显存。

试训通过后，正式训练：

```bash
CUDA_VISIBLE_DEVICES=4,6 FORCE_TORCHRUN=1 "$TCLI" train "$CFG/train.yaml" 2>&1 | tee "$ZBC/logs/train-r0.log"
```

正式训练用两张卡做数据并行：每张卡上各有一份模型，各自处理不同的样本，然后汇总梯度。`FORCE_TORCHRUN=1` 让 LLaMA-Factory 用 torchrun 启动两个进程。
`make_train_configs.py` 默认按 2 张卡生成配置：梯度累积设为 8，所以 2 卡 × 8 = 等效批量 16，和单卡时的设置一致。
如果双卡启动报错，先改用单卡：`CUDA_VISIBLE_DEVICES=4` 加上 `gradient_accumulation_steps=16`，在命令末尾追加这个参数即可，结果是等价的，只是要多花一倍的时间。

主要参数（来自 `configs/train_decision_lora.yaml`）：

| 参数 | 含义 |
|---|---|
| `stage: sft` | 监督微调：让模型输出和专家一样的 JSON 答案 |
| `quantization_bit: 4` + nf4 + 双重量化 | 基座按 4-bit 加载，7B 模型只占约 5 GB 显存 |
| `finetuning_type: lora`，rank 16，alpha 32，`lora_target: all` | 在语言模型的全部线性层上加低秩小矩阵，只训练这些小矩阵 |
| `freeze_vision_tower: true` | 视觉编码器不训练：数据量小，训练它容易过拟合 |
| batch 1 × 梯度累积 16 | 每处理 16 条样本更新一次参数，等效批量 16 |
| `learning_rate: 1e-4`，cosine 衰减，5% 预热 | LoRA 常用的起始学习率，不保证是最优值 |
| `num_train_epochs: 2` | 训练数据过两遍 |
| `cutoff_len: 4096` | 单条样本的 token 上限（正文 1–2k 加上每张图约 144 个 token）；超过会被截断，必须确认答案部分没有被截掉 |
| `bf16`、`gradient_checkpointing` | 半精度计算；不保存中间激活、反向时重新算一遍，用计算时间换显存 |
| `save_steps / eval_steps: 500` | 每 500 步保存一次 checkpoint，并在验证集上算一次 loss |

总步数约为 ⌈训练样本数 / 16⌉ × 2（等效批量 16）。先看日志里前几步每步用了多少秒，再乘以总步数估算总时长。
要看的东西：训练 loss 应该平稳下降；验证 loss 先降，降不动或者开始回升，就说明过拟合了，下一轮可以减少 epoch。

中断后从 checkpoint 接着训：

```bash
CUDA_VISIBLE_DEVICES=4,6 FORCE_TORCHRUN=1 "$TCLI" train "$CFG/train.yaml" resume_from_checkpoint="$ZBC/workspace/lora/r0/checkpoint-500"
```

checkpoint 里存了优化器和学习率调度的状态，所以能接着训。续训时数据和配置都不能改。

## ⑪ 合并 LoRA

```bash
free -h; df -h "$ZBC"
CUDA_VISIBLE_DEVICES= "$TCLI" export "$CFG/export.yaml"
```

在 CPU 上把 LoRA 加回原版 bf16 权重，得到一个完整的新模型，存到 `weights/Qwen2.5-VL-7B-nav-r0`。
合并必须基于原版 bf16 权重，不能在 4-bit 加载的模型上做，所以导出配置里没有 `quantization_bit`。
这一步需要几十 GB 内存和约 16 GB 磁盘。`CUDA_VISIBLE_DEVICES=` 后面留空，表示不用任何 GPU。

## ⑫ 微调后考试和闭环

终端 A（`GPU=6 source ...native_env.sh`）：

```bash
API_HOST=127.0.0.1 API_PORT=18080 API_MODEL_NAME=zbc-r0-nav "$TCLI" api "$ZBC/workspace/train-configs/r0/api-nav.yaml"
```

终端 B：

```bash
$HRUN scripts/eval_decision_offline.py --data "$ZBC/workspace/decision_data/r0-split" \
  --cfg "$CFG/agent-nav.json" --limit 1000000

$HRUN scripts/run_rollout.py --data-root "$DATA" --source-root "$SRC" --splits val_unseen --limit 20 --gpu-ids 0 \
  --cfg "$CFG/agent-nav.json" --plans "$ZBC/workspace/plans/val.jsonl" \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --local-metrics --output-dir "$ZBC/workspace/runs/r0-nav-20"
```

- 微调模型和原版模型的服务名不同、缓存目录也分开（`cache/llm-r0-base` 和 `cache/llm-r0-nav`）。缓存的键里包含模型名，所以微调模型不会读到原版模型的旧答案。
- 对比时保持同样的 20 条路线、同一份 plans、同样的 4-bit 推理，只换模型。
- 离线准确率提高了，不一定代表成功率也会提高，要以闭环 rollout 的结果为准。
- 20 条没有退步，再跑全量：去掉 `--limit 20`，加上 `--splits val_seen val_unseen`，换一个新的输出目录。先用 `compile_instructions.py --splits val_seen val_unseen` 把全量指令补拆好（已经拆过的会自动跳过）。
- 验证集只能拿来评估，不能用来训练：不要在 val 上采数据，也不要用 val 的参考路径做任何调参之外的事。

关于行人：行人按墙钟时间动 12 秒。本地评测时模型每次思考要花几秒，这段时间里行人也在动；而官方是按你提交的动作序列重放，没有这些等待时间。
所以本地算出的碰撞数和官方的可能有出入，参数要以官方回传的分数为准。

## ⑬ DAgger 第 1 轮

保持 r0-nav 服务运行：

```bash
$HRUN scripts/gen_decision_data.py --data-root "$DATA" --limit 1500 --seed 1 --list-episodes \
  --out "$ZBC/workspace/plans/train-r1-ids.txt"
$HRUN scripts/compile_instructions.py --data-root "$DATA" --splits train --workers 1 \
  --episode-list "$ZBC/workspace/plans/train-r1-ids.txt" --cfg "$CFG/agent-base.json" \
  --out "$ZBC/workspace/plans/train.jsonl"
```

拆指令用的是 `agent-base.json`。这时如果服务里跑的是 nav 模型，可以先停掉、换成 base 来拆；更省事的做法是拆指令就一直用 nav 模型，因为指令拆解不属于训练内容，用哪个模型拆都行，只要训练和推理用的是同一份 plans。
已经拆过的 episode 会被跳过。

```bash
$HRUN scripts/gen_decision_data.py --data-root "$DATA" --source-root "$SRC" \
  --plans "$ZBC/workspace/plans/train.jsonl" --cfg "$CFG/agent-nav.json" \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --gpu 0 \
  --dagger --limit 1500 --seed 1 --out "$ZBC/workspace/decision_data/r1"
```

`--dagger` 的意思是：由当前的微调模型来走，专家只负责在每个决策点上写正确答案。
这样收集到的是"模型自己会走到的状态"。第 0 轮的数据都是专家走出来的路，模型没见过自己犯错之后的局面；这一轮专门补上这部分，治"一走偏就回不来"的问题。
这一轮不做随机扰动，因为模型自己就会走错。换 seed 是为了抽到另一批 episode。

采完之后停掉服务，合并两轮数据，并**沿用第 0 轮的验证集**：

```bash
$HRUN scripts/gen_decision_data.py --merge "$ZBC/workspace/decision_data/r0" "$ZBC/workspace/decision_data/r1" \
  --reference "$ZBC/workspace/decision_data/r0-split" --out "$ZBC/workspace/decision_data/r1-split"
"$TPY" scripts/make_train_configs.py --run r1 --root "$ZBC" --dataset "$ZBC/workspace/decision_data/r1-split"
export CFG=$ZBC/workspace/train-configs/r1
CUDA_VISIBLE_DEVICES=4 "$TCLI" train "$CFG/smoke.yaml" && \
  CUDA_VISIBLE_DEVICES=4,6 FORCE_TORCHRUN=1 "$TCLI" train "$CFG/train.yaml" 2>&1 | tee "$ZBC/logs/train-r1.log"
CUDA_VISIBLE_DEVICES= "$TCLI" export "$CFG/export.yaml"
```

- `--reference`：验证场景和验证题都原样沿用 r0 的。新采集到的、属于验证场景的样本全部丢掉，不进训练集。这样每一轮都考同一份卷子，分数才能横向比较。
- r1 是从原版基座出发，在 r0 + r1 的全部数据上重新训练一个 LoRA，而不是在 r0 的 LoRA 基础上接着训。这样结果更好解释，也不会把 r0 的偏差带进来。
- 然后用 r1 的 `api-nav.yaml` 和 `agent-nav.json` 重做 ⑫。一般做 2 轮 DAgger 就够了；如果某一轮闭环分数不再提高，就停止。

## 提交

```bash
$HRUN scripts/make_submission.py --run-dir "$ZBC/workspace/runs/<全量那次>" --splits val_seen val_unseen --data-root "$DATA"
```

脚本会检查每条 episode 的动作是否合法（不超过 500 步、以 STOP 结尾、只用 6 种动作）。如果运行目录里有专家诊断的标记文件，脚本会拒绝打包。

## 出问题先看哪里

| 现象 | 先查 |
|---|---|
| check_env 导入失败 | ③ 的配置路径和 Habitat 补丁 |
| 拆指令 fallback 很多 | 服务是否正常；用 curl 发一条请求，看模型输出的是不是 JSON |
| 离线考试 `json_ok` 低 | 原版模型属于正常现象；训练后仍然低，就检查训练数据里答案是否被截断 |
| 专家 rollout 也很差 | 先不要训练：看 `candidate_coverage`、碰撞率，以及失败 episode 的事件日志 |
| 训练 OOM | 服务有没有停；图片数和正文长度 |
| 闭环分数比离线考试的提升小很多 | 正常现象：离线题都是专家走到的状态，闭环里模型会走到没见过的状态，这正是 ⑬ 要解决的 |
