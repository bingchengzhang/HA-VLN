"""在官方环境里跑智能体：按场景分片、可断点续跑、导出动作序列，可选本地打分。

分片方式与官方 CMA 导出脚本一致（split-scan 为一片），已完成的分片直接跳过。
本地打分时额外记录两样特权信息用于事后归因（不影响动作）：
  ever_within_goal  路上是否曾进入终点 3 m 内
  decision_regret   有多少个决策点选了明显偏离参考路线的候选
"""
import gzip
from collections import Counter
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np

from .adapters.habitat_env import make_env, EpisodeRunner, PrivilegedView, ACTIONS, sensor_report
from .agent import HierarchicalAgent
from .config import AgentCfg
from .metrics import episode_metrics, load_allowed
from .oracle import OracleDecider


def load_episodes(data_root, split):
    path = Path(data_root) / f"HA-R2R/{split}/{split}.json.gz"
    with gzip.open(path, "rt") as f:
        eps = json.load(f)["episodes"]
    eps.sort(key=lambda e: int(e["episode_id"]))
    return eps


def make_jobs(episodes, split, episode_list=None, limit=None):
    if episode_list:
        keep = set(map(str, episode_list))
        episodes = [e for e in episodes if str(e["episode_id"]) in keep]
    if limit:
        episodes = episodes[:limit]
    scans = {}
    for e in episodes:
        scan = Path(e["scene_id"]).parent.name
        scans.setdefault(scan, []).append(str(e["episode_id"]))
    return [{"key": f"{split}-{scan}", "split": split, "ids": ids} for scan, ids in sorted(scans.items())]


def check_actions(actions):
    ok = (1 <= len(actions) <= 500 and all(a in ACTIONS for a in actions)
          and "STOP" not in actions[:-1] and (len(actions) == 500 or actions[-1] == "STOP"))
    if not ok:
        raise ValueError("动作序列不符合官方格式")


def _min_dist_to_polyline(x, y, pts):
    p = np.asarray(pts)
    return float(np.min(np.hypot(p[:, 0] - x, p[:, 1] - y)))


def decision_regret(log, ref_path, thr=1.0) -> int:
    """事后用参考路线评判每个决策：选中的终点比最好的候选离参考路线远 thr 米以上，记一次。"""
    n = 0
    for d in log["decisions"]:
        if d["choose"] is None or not d["endpoints"]:
            continue
        dist = [_min_dist_to_polyline(ex, ey, ref_path) for ex, ey in d["endpoints"]]
        if dist[d["choose"]] - min(dist) > thr:
            n += 1
    return n


def decision_diagnostics(log) -> dict:
    """每个 episode 的决策诊断：候选覆盖、与专家一致、观察次数。"""
    ds = [d for d in log["decisions"] if "diag" in d]
    sem = [d for d in ds if d["source"] in ("llm", "fallback", "expert")]
    agree = [d for d in sem if not d["stop"] and d["diag"]["expert_choose"] is not None]
    return {"n_decisions": len(ds),
            "n_covered": sum(d["diag"]["covered"] for d in ds),
            "n_choice_checked": len(agree),
            "n_choice_agree": sum(d["choose"] == d["diag"]["expert_choose"] for d in agree),
            "n_observe": sum(bool(d.get("observe")) for d in log["decisions"]),
            "n_stop_rejected": sum(e["event"] == "STOP_REJECTED" for e in log["events"]),
            "stop_kind": next((e["event"] for e in reversed(log["events"])
                               if e["event"] in ("STOP_MODEL", "STOP_CONFIRMED", "STOP_UNCONFIRMED", "STOP_NO_EVIDENCE", "BUDGET")), None),
            "n_llm_cached": sum(bool(d.get("cached")) for d in log["decisions"])}


def run_job(job, agent: HierarchicalAgent, args, out_dir: Path, log_dir: Path):
    shard = out_dir / "shards" / (job["key"] + ".json")
    if shard.exists():
        return json.loads(shard.read_text())["rows"]
    env = make_env(args.data_root, job["split"], job["ids"], args.gpu, args.source_root,
                   with_metrics=args.local_metrics)
    rows = []
    try:
        if not (out_dir / "sensor_report.json").exists():
            (out_dir / "sensor_report.json").write_text(json.dumps(sensor_report(env), indent=2, default=str))
        episodes = sorted(env.episodes, key=lambda e: int(e.episode_id))
        env.habitat_env.episode_iterator = iter(episodes)
        runner, priv = EpisodeRunner(env), PrivilegedView(env)
        for ep in episodes:
            obs = runner.reset()
            if str(env.current_episode.episode_id) != str(ep.episode_id):
                raise RuntimeError("reset 到了错误的 episode")
            oracle = None
            if args.local_metrics or args.oracle_decisions:
                oracle = OracleDecider(agent.cfg, priv.reference_path_xyh(), priv.goal_xyh())
            agent.diag_oracle = oracle if args.local_metrics else None
            agent.decision_override = (lambda a, c, ctx, o=oracle: o.label(a, c, ctx["pose"])) \
                if args.oracle_decisions else None
            agent.reset(ep.episode_id, EpisodeRunner.instruction(obs, ep), runner.pose(), runner.height())
            trace, done, info, within = [], False, {}, False
            prev_coll, coll_steps = 0, []
            for _ in range(500):
                action = agent.act(obs["rgb"], obs["depth"], runner.pose(), runner.now(), runner.height())
                trace.append(action)
                obs, done, info = runner.step(action)
                n_coll = int((info.get("collisions_detail") or {}).get("count", 0))
                if n_coll > prev_coll:                      # 累计碰撞数增加：这一步撞了
                    coll_steps.append(len(trace) - 1)
                    prev_coll = n_coll
                if args.local_metrics and priv.distance_to_goal() < 3.0:
                    within = True
                if done:
                    break
            check_actions(trace)
            row = {"episode_id": str(ep.episode_id), "actions": trace}
            if args.local_metrics:
                row.update({"distance_to_goal": float(info.get("distance_to_goal", priv.distance_to_goal())),
                            "collisions": int((info.get("collisions_detail") or {}).get("count", 0)),
                            "stopped": trace[-1] == "STOP", "ever_within_goal": within,
                            "collision_steps": coll_steps,
                            "decision_regret": decision_regret(agent.log, priv.reference_path_xy()),
                            "n_llm_decisions": sum(d["source"] == "llm" for d in agent.log["decisions"]),
                            "wrong_floor": abs(runner.height() - priv.goal_xyh()[2]) > 1.0,
                            **decision_diagnostics(agent.log)})
            rows.append(row)
            (log_dir / f"{ep.episode_id}.json").write_text(json.dumps(
                {**agent.log, **{k: v for k, v in row.items() if k != "actions"}}, ensure_ascii=False, default=float))
    finally:
        env.close()
    tmp = shard.with_suffix(".partial")
    tmp.write_text(json.dumps({"rows": rows}))
    os.replace(tmp, shard)
    return rows


def summarize(out_dir: Path, split: str, allowed_path: str):
    from .metrics import aggregate, failure_table
    rows = []
    for p in sorted((out_dir / "shards").glob(f"{split}-*.json")):
        rows += json.loads(p.read_text())["rows"]
    rows = [r for r in rows if "distance_to_goal" in r]
    if not rows:
        return None
    allowed = load_allowed(allowed_path)
    per_ep = [episode_metrics(r, allowed) for r in rows]
    summary = aggregate(per_ep, split)
    summary["failures"] = failure_table(rows, per_ep)
    summary["llm_decisions_per_ep"] = float(np.mean([r.get("n_llm_decisions", 0) for r in rows]))
    tot = lambda k: sum(r.get(k, 0) for r in rows)
    summary["diagnostics"] = {
        "candidate_coverage": tot("n_covered") / max(tot("n_decisions"), 1),     # 正确方向在候选里的比例：慢回路的上限
        "choice_agreement": tot("n_choice_agree") / max(tot("n_choice_checked"), 1),  # 选择与专家一致的比例
        "false_stop_rate": float(np.mean([r["stopped"] and r["distance_to_goal"] >= 3.0 for r in rows])),
        "wrong_floor_rate": float(np.mean([bool(r.get("wrong_floor")) for r in rows])),
        "observe_per_ep": tot("n_observe") / len(rows),
        "stop_rejected_per_ep": tot("n_stop_rejected") / len(rows),
        "stop_kinds": dict(Counter(r.get("stop_kind") for r in rows)),   # 确认停 / 未确认停 / 步数用完
    }
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
