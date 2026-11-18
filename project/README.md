# HA-VLN

RoboWorld 2026 Track 2 的分层导航智能体。几何模块负责看清环境、算出可走路线并卡住不安全的选项；视觉语言模型只在真正要选择的时候挑一条路、判断子目标是否完成、决定何时停。

系统模块见 [架构](docs/ARCHITECTURE.md)，采集和回测见 [闭环](docs/CLOSED_LOOP.md)，权重见 [模型](docs/MODELS.md)。共享服务器上不用 Docker 的步骤见 [训练操作](docs/TRAINING.md)。楼梯、多楼层和停止确认也写在架构说明里。

命令在本目录执行。数据、权重、运行结果和缓存不进 git。

r0 闭环评估是 20 条路线、SR = 0。训练完成或离线准确率上升，都不能代替这次结果。

## 布局

```
havln_agent/
  config.py                 参数。标了 VERIFY 的用 scripts/check_env.py 对镜像
  geometry.py               habitat (x, y_up, z) ↔ 地图 (x, −z, yaw)
  mapping/
    virtual_laser.py        深度图反投影为虚拟激光
    static_layer.py         每层楼一张高程图，按台阶高度和坡度判断可走
    human_layer.py          行人跟踪和视野外记忆
    costmap.py              墙和人分开膨胀，只算周围 16 m
  perception/human_seg.py   YOLO-seg 人体实例分割
  planning/
    grid_search.py          0.1 m 栅格上一次 Dijkstra
    candidates.py           候选方向和风险摘要
    decision.py             决策点与调用门控
  slow/
    llm_client.py           OpenAI 兼容接口，磁盘缓存
    prompts.py              指令拆解和决策，输出固定 JSON
    slow_loop.py            校验、重试、兜底
    view_memory.py          给视野外候选配以前看过的图
  executor.py               路线到离散动作，停止前确认
  agent.py                  act(rgb, depth, pose, t, height) → 动作
  adapters/habitat_env.py   按官方导出脚本建环境；特权信息单独封装
  oracle.py                 特权专家，只用于标签和诊断
  rollout.py / metrics.py   分片续跑、本地打分、失败归因
scripts/                    权重、环境、rollout、数据、训练、提交
configs/                    QLoRA、合并、DeepSeek 覆盖
tests/                      不依赖 Habitat 的单元测试和合成走廊
```

## 运行

宿主机（只写 `~/zbc`）：

```bash
cd ~/zbc && git clone <仓库> havln_nav
bash havln_nav/scripts/fetch_weights.sh
GPU=1 bash havln_nav/scripts/serve_vlm.sh
docker network connect zbc-net zbc-havln
```

`fetch_weights.sh` 把权重下到 `~/zbc/weights`。`serve_vlm.sh` 在独立容器里起慢回路，`GPU` 换成分配到的卡。Habitat 容器要能访问 `http://zbc-vlm:8000`。

Habitat 容器里把 `~/zbc/havln_nav` 挂到 `/workspace/havln_nav`，`~/zbc/weights` 挂到 `/weights`：

```bash
cd /workspace/havln_nav
bash scripts/setup_env.sh
python scripts/check_env.py

# 20 条冒烟：不调模型，本地打分
python scripts/run_rollout.py --splits val_unseen --limit 20 --no-llm --local-metrics \
    --seg-weights /weights/yolov8m-seg.pt --output-dir /workspace/runs/smoke

python scripts/compile_instructions.py --splits val_seen val_unseen --out /workspace/plans/val.jsonl
python scripts/download_train.py && python scripts/compile_instructions.py --fit-path-stats --out /workspace/plans/path_stats.json
python scripts/run_rollout.py --splits val_seen val_unseen --plans /workspace/plans/val.jsonl \
    --path-stats /workspace/plans/path_stats.json --seg-weights /weights/yolov8m-seg.pt \
    --local-metrics --output-dir /workspace/runs/v1
python scripts/make_submission.py --run-dir /workspace/runs/v1
```

`check_env.py` 对上 `config.py` 里标了 VERIFY 的项，不一致就改配置。`/workspace/runs/v1/metrics_val_unseen.json` 里有本地 SR、NE、CR、TCR、Score 和失败归因。`logs/` 下每个 episode 一份决策记录。

改用 DeepSeek：`export DEEPSEEK_API_KEY=...`，命令加 `--cfg configs/deepseek_text.json`。

## 诊断

同一批 episode 分别用模型和特权专家决策。专家结果只能诊断，`make_submission.py` 会拒绝提交。

```bash
python scripts/run_rollout.py --splits val_unseen --limit 200 --local-metrics --output-dir /workspace/runs/llm
python scripts/run_rollout.py --splits val_unseen --limit 200 --oracle-decisions --output-dir /workspace/runs/oracle
```

`metrics_val_unseen.json` 的 `diagnostics`：

| 指标 | 含义 | 偏低时先看 |
|---|---|---|
| candidate_coverage | 正确方向在候选里的比例 | 候选生成或建图。微调解决不了 |
| choice_agreement | 模型选择与专家一致的比例 | 慢回路 |
| false_stop_rate | 停了但不在 3 m 内 | 停止判断和 goal_box |
| stop_kinds | STOP_CONFIRMED / STOP_UNCONFIRMED / STOP_NO_EVIDENCE / BUDGET | UNCONFIRMED 多：模型过早说停，或深度确认过严 |
| stop_rejected_per_ep | 每条 episode 被驳回的停止次数 | 模型过早喊停 |
| wrong_floor_rate | 结束时和终点不在同一层 | 上下楼选择或楼梯建图 |

专家版 SR 是决策正确时的参考上限，不是严格上界。两者差得多，再微调慢回路。专家版本身就低，先修候选和建图。

## 数据与训练

| 数据 | 来源 | 用途 |
|---|---|---|
| HA-R2R 训练集 10,819 条 | `scripts/download_train.py` | 唯一训练来源。val/test 不用于训练 |
| 训练集子目标 | `compile_instructions.py --splits train` | 与推理相同的 prompt 结构 |
| Qwen2.5-VL-7B 原版 bf16 | `fetch_weights.sh --full-vlm` | QLoRA 基座。AWQ 版不能训练 |
| 决策样本 | `gen_decision_data.py` | 标签见 `havln_agent/oracle.py` |

第 0 轮约 2000 条 episode，大约 1–1.5 万个决策样本。DAgger 每轮再加 1000–1500 条。标签分三类：选候选、停止（带目标框）、转身看。子目标完成是弱标签，按参考路径完成比例估计。

```bash
python scripts/compile_instructions.py --splits train --out /workspace/plans/train.jsonl
python scripts/gen_decision_data.py --plans /workspace/plans/train.jsonl --out /workspace/decision_data/r0 --limit 2000
python scripts/gen_decision_data.py --merge /workspace/decision_data/r0 --out /workspace/decision_data/merged

bash scripts/setup_train_env.sh
GPU=1 bash scripts/train_decision.sh

MODEL_DIR=Qwen2.5-VL-7B-nav MEM=0.85 GPU=1 bash scripts/serve_vlm.sh
python scripts/eval_decision_offline.py --data /workspace/decision_data/merged

python scripts/gen_decision_data.py --plans /workspace/plans/train.jsonl --out /workspace/decision_data/r1 --limit 1500 --dagger --seed 1
python scripts/gen_decision_data.py --merge /workspace/decision_data/r0 /workspace/decision_data/r1 --out /workspace/decision_data/merged
GPU=1 bash scripts/train_decision.sh
```

`--limit 2000` 那次采集里，15% 的可选决策点会故意走错，标签仍是专家答案，用来覆盖走偏后回来的状态。训练和推理共用同一份 prompt、同样的候选标注图和 336×336 分辨率，`goal_box` 坐标系因此一致。

离线评测输出 `json_ok`、`choose_acc`、`stop_recall`、`stop_false`、`observe_match` 和框中心误差。服务器上的分步命令、GPU 分工和版本钉死见 `docs/TRAINING.md`。固定阶段入口是 `scripts/stage.py`，顺序见 `docs/CLOSED_LOOP.md`。

## 已核对的设定

- 行人运动 12 秒后静止（120 帧，10 Hz）
- 碰撞计数来自 `sim.previous_step_collided`，公开代码里擦墙也算。官方计分是否只计人仍要在镜像里核对
- 成功距离 3 m，必须主动 STOP；深度 256×256，RGB 224×224，HFOV 90
- 训练集不在官方下载清单里，`download_train.py` 按同一 Hugging Face 提交拉取

细节和来源见架构说明。

## 已知简化

位姿直接读仿真器，是否允许还待确认。人体检测用现成 YOLO-seg，不微调。不处理镜子和玻璃造成的网格空洞。没有参数搜索，也没有把建图结果对到导航网格的评估脚本。

## 测试

```bash
python -m pip install numpy scipy pillow pytest ruff
python -m ruff check . && python -m ruff format --check .
python -m pytest tests -q
python3 tests/eval_gate.py
```

`pytest` 覆盖建图、走廊端到端、停止回归、训练样本格式和慢回路 HTTP。`eval_gate.py` 打印五个合成场景的调用次数和原因。这些场景只检查代码路径，不代表 Habitat 里的结果。不含人体分割和大模型时，`act()` 每步大约 25–40 ms。
