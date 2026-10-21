"""生成慢回路（Qwen2.5-VL）的微调数据。只在训练集上跑，标签来自特权专家（havln_agent/oracle.py）。

两种模式：
  专家驾驶（第 0 轮）：决策点由专家决定，按 --perturb 的概率故意执行一个错的候选，
                       让数据里有"走偏后怎么回来"的状态。标签永远是专家答案。
  DAgger（第 k 轮）  ：加 --dagger，决策点由当前模型（vLLM 服务里的微调版）决定并执行，
                       专家只打标签。收集的是模型自己会走到的状态，专治"一偏就回不来"。

送给模型的 prompt、图片（含候选编号标注、视野外候选的旧帧）、图片分辨率与推理时完全一致：
共用 SlowLoop.build_text、build_images 和 llm_client.prepare_image。

输出为 LLaMA-Factory 的 sharegpt 多模态格式，按场景划出 5% 做验证集（同一场景的帧不会同时出现在两边）。
可断点续跑：每个场景分片单独存一个 jsonl，已完成的跳过。

  # 第 0 轮：2000 条训练 episode
  python scripts/gen_decision_data.py --plans /workspace/plans/train.jsonl --out /workspace/decision_data/r0 --limit 2000
  # 第 1 轮 DAgger：先用 serve_vlm.sh 起好第 0 轮微调后的模型
  python scripts/gen_decision_data.py --plans /workspace/plans/train.jsonl --out /workspace/decision_data/r1 \\
      --limit 1500 --dagger --seed 1
  # 合并多轮数据并划分
  python scripts/gen_decision_data.py --merge /workspace/decision_data/r0 /workspace/decision_data/r1 \\
      --out /workspace/decision_data/merged
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from havln_agent.agent import HierarchicalAgent
from havln_agent.config import AgentCfg
from havln_agent.oracle import OracleDecider
from havln_agent.slow.llm_client import ChatClient, prepare_image
from havln_agent.slow.prompts import DECIDE_SYSTEM
from havln_agent.slow.slow_loop import SlowLoop, PlanStore, Decision
from havln_agent.slow.view_memory import build_images


class Recorder:
    """决策点钩子：专家打标签、写样本，再决定这一步实际执行什么。"""

    def __init__(self, cfg: AgentCfg, out: Path, perturb: float, dagger: bool, rng: random.Random):
        self.cfg, self.out, self.perturb, self.dagger, self.rng = cfg, out, perturb, dagger, rng
        (out / "images").mkdir(parents=True, exist_ok=True)
        self.oracle = None
        self.episode_id, self.scan, self.k = None, None, 0
        self.rows = []

    def new_episode(self, oracle: OracleDecider, episode_id, scan):
        self.oracle, self.episode_id, self.scan, self.k = oracle, str(episode_id), scan, 0

    def __call__(self, agent: HierarchicalAgent, cands, ctx) -> Decision:
        c = self.cfg
        pose = ctx["pose"]
        label = self.oracle.label(agent, cands, pose)
        allow_observe = not agent.observed_last and agent.n_observes < c.decision.max_observes
        if label.observe and not allow_observe:            # 与推理时同一条规则：不允许看时专家也不能要求看
            label.observe = None

        images, captions = build_images(ctx["rgb"], pose, cands, agent.memory, c.sensor.hfov_deg, floor=agent.floor)
        text = agent.slow.build_text(agent.instruction, agent.plan, agent.sub_idx, cands,
                                     agent.humans.describe(pose, agent.feet), agent.step,
                                     agent.trig.st.traveled_m, captions, True, agent.note)
        self._valid_ids = [k.cid for k in cands]
        valid = label.choose is not None or label.stop or label.observe
        if valid:
            self._write(text, images, captions, label)

        if self.dagger:                                     # 当前模型决定实际走哪
            return agent.slow.decide(agent.instruction, agent.plan, agent.sub_idx, cands,
                                     agent.humans.describe(pose, agent.feet), agent.step,
                                     agent.trig.st.traveled_m, images, captions,
                                     prev_yaw=agent.prev_yaw, allow_observe=allow_observe, note=agent.note)
        viable = [k for k in cands if k.path is not None]
        if not label.stop and not label.observe and len(viable) > 1 and self.rng.random() < self.perturb:
            wrong = self.rng.choice([k.cid for k in viable if k.cid != label.choose])
            return Decision(wrong, label.subgoal_done, False, "none", "perturbed", "expert")
        return label

    def _write(self, text, images, captions, label: Decision):
        paths = []
        for i, img in enumerate(images):
            rel = f"images/{self.episode_id}_{self.k:03d}_{i}.jpg"
            prepare_image(img, self.cfg.slow.image_side).save(self.out / rel, quality=88)
            paths.append(rel)
        # 与推理时消息结构一致：正文，然后"说明 + 图片"交替
        user = text + "\n" + "\n".join(f"{cap}\n<image>" for cap in captions)
        answer = dict(label.to_json(), reason="follow the instruction")
        self.rows.append({"scan": self.scan, "episode_id": self.episode_id,
                          "raw": {"text": text, "captions": captions, "label": label.to_json(),
                                  "valid_ids": self._valid_ids},
                          "messages": [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}], "images": paths})
        self.k += 1


def _scene(r):
    return r["scan"].split("-", 1)[-1]


def write_split(sources, out: Path, val_frac=0.05, seed=0, reference=None):
    """合并若干轮数据目录，按场景划分训练/验证，写 LLaMA-Factory 的 dataset_info.json。

    - 图片路径一律写成绝对路径：合并目录只放 json，图片留在各轮的采集目录里（所以采集目录不能删）。
    - 每条样本检查：图片文件存在、<image> 占位符个数与图片数一致，不一致直接报错，不带病训练。
    - reference：DAgger 轮次传入第 0 轮的划分目录。验证场景沿用它的 heldout_scans.json，
      验证题也原样沿用它的题目——每轮考同一张卷子，分数才能横向比；新采集的、落在验证场景里的样本全部丢弃。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in map(lambda x: Path(x).resolve(), sources):
        for f in sorted((src / "shards").glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["images"] = [str(src / p) for p in r["images"]]
                missing = [p for p in r["images"] if not Path(p).is_file()]
                if missing:
                    raise SystemExit(f"{f} 缺图片：{missing[0]}")
                n_ph = sum(m["content"].count("<image>") for m in r["messages"])
                if n_ph != len(r["images"]):
                    raise SystemExit(f"{f} 占位符 {n_ph} 个、图片 {len(r['images'])} 张，对不上")
                rows.append(r)
    if not rows:
        raise SystemExit("没有找到任何样本（shards/*.jsonl 为空）")

    if reference:
        ref = Path(reference)
        held = set(json.loads((ref / "heldout_scans.json").read_text()))
        val_rows = json.loads((ref / "decision_val_full.json").read_text())
    else:
        scans = sorted({_scene(r) for r in rows})
        if len(scans) < 2:
            raise SystemExit("样本只来自一个场景，无法按场景留出验证集；多采几条 episode 再划分")
        random.Random(seed).shuffle(scans)
        held = set(scans[:max(1, int(len(scans) * val_frac))])
        val_rows = [r for r in rows if _scene(r) in held]
    train_rows = [r for r in rows if _scene(r) not in held]
    if not train_rows or not val_rows:
        raise SystemExit("训练或验证为空")

    strip = lambda r: {"messages": r["messages"], "images": r["images"]}
    dump = lambda name, v: (out / name).write_text(json.dumps(v, ensure_ascii=False))
    dump("decision_train.json", [strip(r) for r in train_rows])
    dump("decision_val.json", [strip(r) for r in val_rows])
    dump("decision_val_full.json", val_rows)                         # 下一轮 --reference 用
    # 离线评测用：原始 prompt 正文、图片说明和标签（scripts/eval_decision_offline.py）
    dump("decision_val_raw.json", [dict(r["raw"], images=r["images"]) for r in val_rows if "raw" in r])
    (out / "heldout_scans.json").write_text(json.dumps(sorted(held), indent=2))
    info = {name: {"file_name": f"{name}.json", "formatting": "sharegpt",
                   "columns": {"messages": "messages", "images": "images"},
                   "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                            "assistant_tag": "assistant", "system_tag": "system"}}
            for name in ("decision_train", "decision_val")}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    kinds = {"choose": 0, "stop": 0, "observe": 0}
    for r in train_rows:
        a = json.loads(r["messages"][-1]["content"])
        kinds["stop" if a["stop"] else "observe" if a["observe"] else "choose"] += 1
    manifest = {"sources": [str(x) for x in sources], "reference": str(reference) if reference else None,
                "seed": seed, "train": len(train_rows), "val": len(val_rows),
                "heldout_scans": sorted(held), "train_label_kinds": kinds}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"训练 {len(train_rows)}，验证 {len(val_rows)}（验证场景 {len(held)} 个"
          f"{'，沿用参考划分' if reference else ''}）；训练标签分布 {kinds}")


def sample_episodes(data_root, limit, seed):
    """与正式采集完全相同的抽样：同一个 seed 抽到同一批 episode。"""
    from havln_agent.rollout import load_episodes
    rng = random.Random(seed)
    eps = load_episodes(data_root, "train")
    rng.shuffle(eps)
    return eps[:limit], rng


def main():
    from havln_agent.adapters.habitat_env import make_env, EpisodeRunner, PrivilegedView
    from havln_agent.perception.human_seg import build_segmenter
    from havln_agent.rollout import make_jobs

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/havln2")
    p.add_argument("--source-root", default="/workspace/cma-deps/HA-VLN")
    p.add_argument("--plans", help="训练集子目标缓存（compile_instructions.py --splits train）")
    p.add_argument("--out", required=True)
    p.add_argument("--cfg", default=None)
    p.add_argument("--limit", type=int, default=2000, help="随机抽多少条训练 episode")
    p.add_argument("--perturb", type=float, default=0.15, help="专家驾驶时故意走错的概率")
    p.add_argument("--dagger", action="store_true", help="由当前模型驾驶、专家打标签（需要 vLLM 服务）")
    p.add_argument("--seed", type=int, default=0, help="每一轮换一个种子，抽到不同的 episode")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seg-weights", default="/weights/yolov8m-seg.pt")
    p.add_argument("--merge", nargs="+", help="只合并这些轮次的数据目录并划分，不跑仿真")
    p.add_argument("--reference", default=None, help="与 --merge 连用：沿用这个划分目录的验证场景和验证题")
    p.add_argument("--list-episodes", action="store_true",
                   help="只把这次会抽到的 episode id 写到 --out（一行一个），给 compile_instructions 预先拆指令用")
    args = p.parse_args()

    out = Path(args.out)
    if args.merge:
        return write_split(args.merge, out, reference=args.reference)
    if args.list_episodes:
        eps, _ = sample_episodes(args.data_root, args.limit, args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(str(e["episode_id"]) for e in eps) + "\n")
        return print(f"{len(eps)} 个 episode id 写到 {out}")

    cfg = AgentCfg.from_json(args.cfg)
    eps, rng = sample_episodes(args.data_root, args.limit, args.seed)
    rec = Recorder(cfg, out, args.perturb, args.dagger, rng)
    seg = build_segmenter("yolo", args.seg_weights, device=f"cuda:{args.gpu}",
                          conf=cfg.human.det_conf, min_pixels=cfg.human.min_mask_pixels)
    client = ChatClient(cfg.slow) if args.dagger else None
    plans = PlanStore(args.plans)
    jobs = make_jobs(eps, "train")
    (out / "shards").mkdir(parents=True, exist_ok=True)

    for job in jobs:
        shard = out / "shards" / f"{job['key']}.jsonl"
        if shard.exists():
            continue
        env = make_env(args.data_root, "train", job["ids"], args.gpu, args.source_root)
        rec.rows = []
        try:
            episodes = sorted(env.episodes, key=lambda e: int(e.episode_id))
            env.habitat_env.episode_iterator = iter(episodes)
            runner, priv = EpisodeRunner(env), PrivilegedView(env)
            agent = HierarchicalAgent(cfg, seg, SlowLoop(cfg, client), plans, decision_override=rec)
            for ep in episodes:
                obs = runner.reset()
                rec.new_episode(OracleDecider(cfg, priv.reference_path_xyh(), priv.goal_xyh()), ep.episode_id, job["key"])
                agent.reset(ep.episode_id, EpisodeRunner.instruction(obs, ep), runner.pose(), runner.height())
                for _ in range(500):
                    action = agent.act(obs["rgb"], obs["depth"], runner.pose(), runner.now(), runner.height())
                    obs, done, _ = runner.step(action)
                    if done:
                        break
        finally:
            env.close()
        tmp = shard.with_suffix(".partial")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rec.rows))
        tmp.replace(shard)
        print(f"{job['key']}：{len(rec.rows)} 个样本", flush=True)

    write_split([out], out)


if __name__ == "__main__":
    main()
