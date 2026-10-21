# 给 Codex 的执行任务（第 2 版，替代旧版 G0–G14）

旧版每关都要现写编排脚本、每关都停。现在选卡、起停模型服务、检查、写报告全部固定在 `scripts/stage.py` 里。
**你的工作只有三件：同步代码、按顺序跑下面的命令、把每个阶段生成的 `report.md` 原样贴回来。**
不要另写编排脚本，不要改项目源码；命令失败时 report.md 里已经写了原因和日志位置，贴回来等确认。

## 硬性规则

1. 服务器 `ffy@10.130.130.66 -p 6066` 是共享机器，写入只落在 `/mnt/data/zbc`。不用 sudo，不改 `~/.bashrc` 等账号配置，不用 Docker。
2. **登录凭据不得以明文出现在任何命令、脚本、日志里。** 用已有的加密凭据文件，或者让我在本机配好 SSH 密钥。
3. GPU：只用 5、6、7 号里此刻空闲的卡。`stage.py` 会自己检查并挑选（`ALLOWED_GPUS` 默认就是 `5,6,7`），卡不够会直接退出。不要手动指定其他卡，不要停任何别人的进程。
4. 验证集只用于评估。`--oracle-decisions` 的结果禁止提交。
5. 长任务用 `nohup ... &` 挂后台，**跑完之前不要轮询**。完成后读一次 `report.md` 即可（`status.json` 里有当前步骤，确实需要时才看）。
6. 不要把大日志、图片整份读进对话。只回传 report.md 和我点名要的文件。

## 0. 同步代码（本机 → 服务器）

项目正本目前在 `/mnt/data/zbc/workspace/training-r0/havln_nav`。先备份，再换成新版：

```bash
# 本机上传新 zip 到 /mnt/data/zbc/installers/havln_nav.zip 后，在服务器执行：
cd /mnt/data/zbc/workspace/training-r0
mv havln_nav "/mnt/data/zbc/backups/havln_nav-$(date +%m%d-%H%M)"
python3 -c "import zipfile; zipfile.ZipFile('/mnt/data/zbc/installers/havln_nav.zip').extractall('.')"
cd havln_nav && GPU=none source scripts/native_env.sh
$HPY tests/test_mapping.py && $HPY tests/test_agent.py && $HPY tests/test_llm_client.py && $HPY tests/test_data_gen.py
```

四个测试都应该全部 PASS（纯 CPU，几分钟）。有 FAIL 就停下，贴输出。
以后每个新终端都先 `cd .../havln_nav && GPU=none source scripts/native_env.sh`。

本版改动（供核对）：
- 俯仰步长默认 30°（与比赛配置一致）；之前的外部覆盖文件 `agent-runtime-r0.json` 不再需要。
- 指令拆解输出上限 1024 token，失败时绕过缓存重试，拆解失败的行重跑时会重新拆。
  **旧的 `workspace/plans/val_unseen.jsonl` 如果存在，先改名备份**，让它用新上限重拆。
- 墙边禁区脱困（之前那条卡在同一位置 67 步的问题）；擦墙滑动记为 SLIDE；rollout 记录 collision_steps。
- check_env 读不到的值不再显示 OK。

## 1. 无模型冒烟（约 20 条）

```bash
nohup $HPY scripts/stage.py smoke --n 20 > $ZBC/logs/stage-smoke.out 2>&1 &
```

完成后贴 report.md，再贴 `diag-nollm.txt` 的前 20 行。失败就停。

## 2. 人体真值掩码冒烟（几分钟，和第 1 步可以并行，用另一张空卡）

```bash
GPU=<5/6/7 里此刻空闲的一张> source scripts/native_env.sh
$HRUN scripts/seg_gt_smoke.py --data-root "$DATA" --source-root "$SRC" --gpu 0 \
  --seg-weights "$ZBC/weights/yolov8m-seg.pt" --out "$ZBC/workspace/diag/seg-smoke"
```

贴最后的“结论”和“钩子统计”两行，以及 `smoke.json` 中 rows 的前 10 行。把 `step*.png` 拷回本机（红色是真值，绿色是 YOLO）。

## 3. 原版 vs 专家（检查点 1，必须停）

```bash
nohup $HPY scripts/stage.py paired --n 20 > $ZBC/logs/stage-paired.out 2>&1 &
```

两张卡：一张跑 Qwen 服务（专家组也在这张卡上并行跑），另一张跑原版组的仿真。HF 后端较慢，预计数小时。
完成后贴 report.md 全文。**到这里停下等我确认**，不要进入第 4 步。

## 以后的顺序（等确认后再执行，届时我会逐条放行）

| 步骤 | 命令 | 停不停 |
|---|---|---|
| 4 试采 | `stage.py collect --n 20 --tag pilot` | **检查点 2**：贴 report.md，拷回 `labels/sheet-*.jpg` |
| 5 正式采集 | `stage.py collect --n 500 --tag r0` | 自动检查不出问题就继续 |
| 6 原版离线考试 | `stage.py offline --run r0 --model base --data r0` | 继续 |
| 7 训练 | `stage.py train --run r0` | 试训失败或 OOM 就停；需要两张空卡 |
| 8 微调后离线考试 | `stage.py offline --run r0 --model nav --data r0` | 继续 |
| 9 微调后闭环 | `stage.py closed --run r0 --model nav --n 20 --compare <第 3 步原版目录>` | **检查点 3**：贴 report.md |

## 汇报格式

```
阶段：
命令：（原样）
report.md：（原样全文）
我额外发现的问题：（没有就写“无”）
```

没执行的写“未执行”，不要把计划写成已完成。
