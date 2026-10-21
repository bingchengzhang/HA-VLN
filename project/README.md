# havln_nav · ExcuseMeBro 分层导航

RoboWorld 2026 Track 2（HA-VLN 2.0）的分层导航智能体，最小闭环版本：能跑通、能提交、能看出错在哪。

系统模块与反馈路径见 [架构说明](docs/ARCHITECTURE.md)，采集、训练和回测流程见 [闭环说明](docs/CLOSED_LOOP.md)，训练权重的发布状态见 [模型说明](docs/MODELS.md)。

原则：大模型只做选择题。快回路负责看清环境、算好每条路；慢回路只在真正有岔路的地方挑一条、判断何时停。
规则只负责剔除物理上不可行的选项和保证安全，涉及"指令想让我去哪"的判断一律留给模型。

## 目录

```
havln_agent/
  config.py              全部参数；标了 VERIFY 的进镜像核对（scripts/check_env.py 会自动比对）
  geometry.py            坐标约定：habitat (x, y_up, z) ↔ 地图 (x, −z, yaw)
  mapping/
    virtual_laser.py     深度图 → 虚拟激光（反投影、俯仰角、人体掩码、截断深度）
    static_layer.py      静态层：每层楼一张 2.5D 高程图，按台阶高度/坡度判断能不能走（楼梯、桌子、断崖）
    human_layer.py       行人层：人体点 + 跟踪 + 视野外记忆
    costmap.py           代价图：墙/人分别膨胀，只算 agent 周围 16 m 窗口
  perception/human_seg.py  现成 YOLO-seg 人体实例分割（顺带输出画面里的物体类别）
  planning/
    grid_search.py       0.1 m 规划栅格 + 一次 Dijkstra 得到到所有格子的路线
    candidates.py        候选方向 + 风险摘要
    decision.py          决策点（路段走完/岔口/路线失效/兜底/走过头）与调用门控
  slow/
    llm_client.py        OpenAI 兼容接口（本地 vLLM 与 DeepSeek 通用），磁盘缓存保证可复现
    prompts.py           指令拆解与决策的 prompt，输出固定 JSON
    slow_loop.py         拆解、决策、输出校验、重试、兜底
    view_memory.py       视野记忆：给视野外的候选配一张以前朝那边看过的图
  executor.py            路线 → 离散动作；紧急覆盖（等人）；STOP 双确认
  agent.py               主循环 act(rgb, depth, pose, t, height) → 动作；分楼层建图、换层后转一圈、前方未知时低头看
  adapters/habitat_env.py  按官方导出脚本的方式建环境；特权信息单独封装（只给评测/造数据用）
  oracle.py              特权专家：训练标签、--oracle-decisions 诊断、候选覆盖率（不进入正式推理）
  rollout.py / metrics.py  分片续跑、本地打分（复现官方公式）、失败归因、决策诊断
scripts/
  fetch_weights.sh       YOLO-seg、Qwen2.5-VL-7B（ModelScope）、可选 ETPNav 权重
  serve_vlm.sh           vLLM 起慢回路服务（独立容器，与 Habitat 镜像依赖隔离）
  setup_env.sh           官方容器里补装依赖，不动 torch/numpy/habitat
  check_env.py           建环境、打印实际传感器参数、核对 VERIFY 项、确认坐标约定
  download_train.py      下载 HA-R2R 训练集（官方工具包只下 val）
  compile_instructions.py  离线拆指令；--fit-path-stats 在训练集拟合路径长度
  run_rollout.py         跑智能体（多卡、可续跑、可本地打分）
  make_submission.py     合并分片、格式检查、打包、官方校验
  gen_decision_data.py   在训练集上生成慢回路微调数据（专家驾驶 / DAgger），--merge 合并划分
  eval_decision_offline.py  不跑仿真，直接在验证样本上评测模型决策
  setup_train_env.sh / train_decision.sh  独立训练环境；QLoRA 微调 + 合并
configs/
  train_decision_lora.yaml  LLaMA-Factory 的 QLoRA 微调配置
  export_merged.yaml        LoRA 合并回原版权重
  deepseek_text.json        改用 DeepSeek API（纯文本）时的参数覆盖
tests/                   不需要 Habitat 的单元测试和合成走廊端到端测试
```

## 代码放哪

GitHub 公开仓库保存代码正本，Hugging Face 保存训练权重。WSL 里改代码、`git push`；服务器拉取仓库后从 `project/` 目录运行。
不在服务器上直接改代码。数据、权重、运行结果、缓存都不进 git（见 `.gitignore`）。

## 从零跑起来

宿主机（服务器，师兄账号下只动 `~/zbc`）：

```bash
cd ~/zbc && git clone <你们的仓库> havln_nav
bash havln_nav/scripts/fetch_weights.sh                 # 权重到 ~/zbc/weights
GPU=1 bash havln_nav/scripts/serve_vlm.sh               # 慢回路服务，GPU 换成指定的卡
docker network connect zbc-net zbc-havln                # 让 Habitat 容器能访问 http://zbc-vlm:8000
```

Habitat 容器里（启动时把 `~/zbc/havln_nav` 挂到 `/workspace/havln_nav`，`~/zbc/weights` 挂到 `/weights`）：

```bash
cd /workspace/havln_nav
bash scripts/setup_env.sh
python scripts/check_env.py                              # 先核对参数，有"不一致"就改 config.py

# 冒烟：20 条，不调大模型，本地打分
python scripts/run_rollout.py --splits val_unseen --limit 20 --no-llm --local-metrics \
    --seg-weights /weights/yolov8m-seg.pt --output-dir /workspace/runs/smoke

# 正式
python scripts/compile_instructions.py --splits val_seen val_unseen --out /workspace/plans/val.jsonl
python scripts/download_train.py && python scripts/compile_instructions.py --fit-path-stats --out /workspace/plans/path_stats.json
python scripts/run_rollout.py --splits val_seen val_unseen --plans /workspace/plans/val.jsonl \
    --path-stats /workspace/plans/path_stats.json --seg-weights /weights/yolov8m-seg.pt \
    --local-metrics --output-dir /workspace/runs/v1
python scripts/make_submission.py --run-dir /workspace/runs/v1
```

`/workspace/runs/v1/metrics_val_unseen.json` 里有本地 SR/NE/CR/TCR/Score 和失败归因分布，
`logs/` 下每个 episode 一份决策记录（每个决策点的候选、风险摘要、选择、理由）。

改用 DeepSeek：`export DEEPSEEK_API_KEY=...`，命令里加 `--cfg configs/deepseek_text.json`。

## 跑通基线之后：诊断、数据与训练

### 1. 先诊断，再决定训不训

```bash
# 同样的前 200 条，分别用大模型和特权专家做决策（专家版只能诊断，make_submission 会拒绝提交它）
python scripts/run_rollout.py --splits val_unseen --limit 200 --local-metrics --output-dir /workspace/runs/llm
python scripts/run_rollout.py --splits val_unseen --limit 200 --oracle-decisions --output-dir /workspace/runs/oracle
```

`metrics_val_unseen.json` 里的 `diagnostics`：

| 指标 | 含义 | 低了说明 |
|---|---|---|
| candidate_coverage | 正确方向在候选里的比例 | 候选生成或建图有问题，训模型没用 |
| choice_agreement | 模型选择与专家一致的比例 | 慢回路是瓶颈，该微调 |
| false_stop_rate | 停了但不在 3 m 内 | 停止判断；看 goal_box 是否准 |
| stop_kinds | 结束方式：STOP_CONFIRMED / STOP_UNCONFIRMED / STOP_NO_EVIDENCE / BUDGET | UNCONFIRMED 多说明模型乱喊"到了"或深度确认太严 |
| stop_rejected_per_ep | 每条 episode 被驳回的停止次数 | 高说明模型过早喊停 |
| wrong_floor_rate | 结束时和终点不在同一层 | 上下楼的选择或楼梯建图 |

专家版的 SR 是"决策完美时"的上限。两者差得多 → 微调慢回路；专家版本身就低 → 先修候选和建图。

### 2. 需要的数据

| 数据 | 来源 | 用途 |
|---|---|---|
| HA-R2R 训练集 10,819 条 | `scripts/download_train.py` | 唯一的训练来源（val/test 禁止用于训练） |
| 训练集的子目标拆解 | `compile_instructions.py --splits train` | 和推理时同样的 prompt 结构 |
| Qwen2.5-VL-7B 原版 bf16 | `fetch_weights.sh --full-vlm` | QLoRA 只能在原版权重上训 |
| 决策样本（专家标签） | `gen_decision_data.py` | 微调数据，标签见 `havln_agent/oracle.py` |

规模建议：第 0 轮 2000 条 episode，约 1–1.5 万个决策样本；DAgger 每轮再加 1000–1500 条 episode。
标签分三类：选候选、停止（带目标框）、转身看。子目标完成是弱标签（按参考路径完成比例估计）。

### 3. 训练流程

```bash
# 容器内：造数据（第 0 轮专家驾驶，15% 的决策点故意走错以覆盖"走偏后怎么回来"）
python scripts/compile_instructions.py --splits train --out /workspace/plans/train.jsonl
python scripts/gen_decision_data.py --plans /workspace/plans/train.jsonl --out /workspace/decision_data/r0 --limit 2000
python scripts/gen_decision_data.py --merge /workspace/decision_data/r0 --out /workspace/decision_data/merged

# 宿主机：建训练环境、微调、合并
bash scripts/setup_train_env.sh
GPU=1 bash scripts/train_decision.sh

# 换上新模型，先离线评测（几分钟），再跑 rollout
MODEL_DIR=Qwen2.5-VL-7B-nav MEM=0.85 GPU=1 bash scripts/serve_vlm.sh
python scripts/eval_decision_offline.py --data /workspace/decision_data/merged

# DAgger：让新模型自己开、专家打标签，合并后再训一轮
python scripts/gen_decision_data.py --plans /workspace/plans/train.jsonl --out /workspace/decision_data/r1 --limit 1500 --dagger --seed 1
python scripts/gen_decision_data.py --merge /workspace/decision_data/r0 /workspace/decision_data/r1 --out /workspace/decision_data/merged
GPU=1 bash scripts/train_decision.sh
```

训练与推理用的是同一份 prompt、同样的图片（候选编号标注、视野外候选的旧帧）、同样的 336×336 分辨率，
`goal_box` 的坐标系也因此一致。离线评测给出 json_ok、choose_acc、stop_recall、stop_false、observe_match、框中心误差。

## 从源码查实的事实

- 行人：120 帧 @ 10 Hz，按墙钟推进，12 秒后静止（`HASimulator/environments.py`）
- 碰撞：行人以静态物体加入场景并重算导航网格（agent 半径 0.1）；碰撞计数是 `sim.previous_step_collided`，
  按公开代码擦墙也算。**VERIFY**：官方计分镜像 `/opt/havln-challenge/scoring` 是否只计人
- TCR = max(0, 碰撞数 − 参考路径上本就避不开的碰撞数)，CR 分母是受人影响的 episode 比例（`HASimulator/metric.py`）
- 成功距离 3 m（三维欧氏距离，楼层错了高度差也算进去），必须主动 STOP；`ALLOW_SLIDING: True`；深度 256×256、RGB 224×224、HFOV 90
- 官方工具包的下载清单里没有训练集；`download_train.py` 按同一 HF 版本号拉取，路径待实测

## 楼梯与多楼层

做法参考机器人领域的高程图可通行性分析，以及 Habitat 多楼层导航工作（MFNP、ASCENT）的分楼层建图：

- 每格存观测到的最低表面（只用低于相机的点）和身体高度内的最高点。格内竖直跨度超过一个台阶（0.2 m）是障碍；
  相邻已知格高差超过"台阶 + tan(38°) × 中间未观测缝宽"也是障碍（标较高的一侧）。楼梯逐级 0.18 m 能走，
  桌子、矮柜、平台边缘不能走。判定每次由当前高程重新算，看得越全越准
- 水平相机看不到脚前、比脚下低的地方：路线前方 1 m 内有未观测格时低头看一眼（两步），下楼梯和断崖靠这个看清
- 脚下高度偏离本层参考高度超过 1.5 m（连续 3 步）就换一层图，新楼层先原地转一圈
- 候选里标 `goes`: up / down / level，大模型据此对应"上楼""下楼"这类指令
- 行人按高度归属楼层，楼上楼下的人互不干扰
- 不需要任何楼梯检测模型。需要在镜像里核实：导航网格最大台阶和坡度、TILT_ANGLE（`check_env.py` 会打印）

## 当前已知的简化（等实测再决定要不要加）

位姿直接读仿真器（待主办方确认）；人体检测用现成 YOLO-seg，不微调；不处理镜子/玻璃造成的网格空洞；
没有参数自动搜索和建图对比导航网格的评估脚本。

## 测试

```bash
python3 tests/test_mapping.py   # 建图：墙、上下楼梯、桌子/矮柜/门槛、平台边缘、膨胀
python3 tests/test_agent.py     # 端到端：L 形走廊、绕行人、上楼到二层、下楼；慢回路解析与兜底
python3 tests/eval_gate.py      # 门控：五个场景的调用次数、间隔与原因
python3 tests/test_data_gen.py  # 训练数据：专家驾驶生成样本，检查格式能被 LLaMA-Factory 读取、标签能被推理端解析
python3 tests/test_llm_client.py  # 慢回路 HTTP：假的 OpenAI 兼容服务，检查请求结构、图片尺寸、缓存
```

合成场景只验证代码逻辑自洽，不代表 Habitat 里的效果。不含人体分割和大模型时，`act()` 每步约 25–40 ms。

## 训练

服务器上的完整训练操作（原生环境、无 Docker）见 `docs/TRAINING.md`。

## 停止规则（模型说"到了"之后）

1. 有目标框就量框内深度，否则量模型说的那一侧；≤ 2.5 m 直接停（STOP_CONFIRMED）。
2. 太远就朝它走最多 16 步，走完或被挡后再量一次。
3. 还是确认不了：不停，记一次 STOP_REJECTED，把原因写进下一次给模型的提示里，回到决策。
4. 连续被驳回超过 `decision.max_stop_rejects`（默认 2）次后模型仍坚持，才按它的判断停（STOP_UNCONFIRMED）——
   永远不停必定失败，所以最后让模型说了算，但会在日志里单独统计。
5. 模型没给任何方位（goal_side=none 且无框）时直接停，记 STOP_NO_EVIDENCE。

模型输出的 stop / subgoal_done 按严格布尔解析，字符串 "false" 不会被当成真；认不出的值整条决策作废、走兜底。
视野记忆按楼层分开存取；转身观察每条 episode 最多 `decision.max_observes` 次。
