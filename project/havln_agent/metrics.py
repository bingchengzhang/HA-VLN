"""本地评测：复现官方计分公式，并给每个失败 episode 归因。

TCR/CR 的扣减规则来自公开源码 HASimulator/metric.py：
  每条 episode 的碰撞数减去该 episode 参考路径上本来就避不开的碰撞数（collision_num_{split}.json）
  CR 的分母是"受人影响的 episode"比例 beta
本地结果只做相对比较，官方以 CodaBench 重放为准。
"""
import json
from collections import Counter

BETA = {"train": 1.0 - 1615.0 / 10819, "val_seen": 1.0 - 96.0 / 778,
        "val_unseen": 1.0 - 246.0 / 1839, "test": 1.0 - 549.0 / 3408}


def load_allowed(path):
    try:
        with open(path) as f:
            return {str(k): int(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


def episode_metrics(row: dict, allowed: dict, success_dist: float = 3.0) -> dict:
    """row 需要：distance_to_goal、collisions（原始计数）、stopped（最后动作是否 STOP）。"""
    tcr = max(0, int(row["collisions"]) - allowed.get(str(row["episode_id"]), 0))
    reached = row["stopped"] and row["distance_to_goal"] < success_dist
    return {"NE": float(row["distance_to_goal"]), "TCR": tcr, "CR": min(tcr, 1),
            "SR": int(reached and tcr == 0), "reached": int(reached)}


def aggregate(per_ep: list, split: str) -> dict:
    n = max(len(per_ep), 1)
    sr = sum(e["SR"] for e in per_ep) / n
    ne = sum(e["NE"] for e in per_ep) / n
    tcr = sum(e["TCR"] for e in per_ep) / n
    cr = min(1.0, sum(e["CR"] for e in per_ep) / (BETA.get(split, 1.0) * n))
    nav = 0.8 * sr + 0.2 * 3.0 / (3.0 + ne)
    social = 0.75 * (1 - cr) + 0.25 / (1 + tcr)
    return {"n": len(per_ep), "SR": sr, "NE": ne, "CR": cr, "TCR": tcr,
            "reach": sum(e["reached"] for e in per_ep) / n,
            "Score": 100 * nav * (0.7 + 0.3 * social)}


def attribute_failure(row: dict, m: dict, success_dist: float = 3.0) -> str:
    """失败归因：每个失败 episode 只打一个主标签，按"最先该修哪里"排序。"""
    if m["SR"]:
        return "success"
    if m["reached"] and m["TCR"] > 0:
        return "collision"                       # 到了但撞过
    if not row["stopped"]:
        return "budget_exhausted"                # 走满 500 步还没停
    if row.get("ever_within_goal"):
        return "stopped_wrong_place"             # 曾经到过目标 3 m 内，最后停错了地方：停止判断问题
    if row.get("decision_regret", 0) > 0:
        return "wrong_choice"                    # 有决策点选了明显偏离参考路线的候选
    return "never_reached"


def failure_table(rows, per_ep) -> dict:
    return dict(Counter(attribute_failure(r, m) for r, m in zip(rows, per_ep)))
