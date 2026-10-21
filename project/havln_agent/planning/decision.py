"""决策点与调用门控。

原则：规则只负责两件事——剔除物理上不可行的选项、保证安全。
凡是涉及"指令想让我去哪"的判断都留给模型，所以门控只在"确实没得选"时跳过调用。

决策点只有四种：
  SEGMENT_DONE   当前路段走完（每段 ≤ segment_max_m）
  JUNCTION       侧面出现了新的开口（以 agent 为圆心扫一圈，可通行缺口数比上一步多）
  PATH_INVALID   当前路线失效，且附近没有同方向的替代
  FALLBACK       距上次决策超过 llm_fallback_steps 步
另有 OVERRUN：走过的距离远超这条指令"应有"的路径长度，说明可能走过头，强制问一次。
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import AgentCfg
from .candidates import Candidate
from .grid_search import PlanGrid

# 指令里的地标词 → YOLO（COCO）类别。检测到对应类别，就说明当前子目标"可能已完成"，值得问一次
LANDMARK_TO_COCO = {
    "sofa": "couch", "couch": "couch", "bed": "bed", "bedroom": "bed", "toilet": "toilet", "bathroom": "toilet",
    "tv": "tv", "television": "tv", "chair": "chair", "table": "dining table", "dining": "dining table",
    "sink": "sink", "fridge": "refrigerator", "refrigerator": "refrigerator", "kitchen": "refrigerator",
    "oven": "oven", "microwave": "microwave", "plant": "potted plant", "clock": "clock", "vase": "vase",
    "laptop": "laptop", "book": "book", "bench": "bench",
}


def count_openings(grid: PlanGrid, pose, radius: float, min_gap_deg: float) -> int:
    """以 agent 为圆心、radius 为半径扫一圈，数连续可通行的角度段有几个（每段至少 min_gap_deg 宽）。"""
    x, y, _ = pose
    n = 72
    free = np.zeros(n, dtype=bool)
    for k in range(n):
        a = 2 * math.pi * k / n
        i, j = grid.to_cell(x + radius * math.cos(a), y + radius * math.sin(a))
        free[k] = bool(grid.inside(i, j) and not grid.wall_lethal[i, j] and not grid.unknown[i, j])
    if free.all():
        return 1
    if not free.any():
        return 0
    start = int(np.argmin(free))
    runs, cur = [], 0
    for s in range(n):
        if free[(start + s) % n]:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    min_len = max(1, int(round(min_gap_deg / (360 / n))))
    return sum(1 for r in runs if r >= min_len)


@dataclass
class DecisionState:
    last_call_step: int = -10 ** 6
    segment_start: tuple = (0.0, 0.0)
    prev_openings: Optional[int] = None
    traveled_m: float = 0.0
    last_xy: Optional[tuple] = None


class DecisionTrigger:
    def __init__(self, cfg: AgentCfg, expected_path_m: Optional[float] = None):
        self.cfg = cfg
        self.st = DecisionState()
        self.expected_path_m = expected_path_m
        self._overrun_fired = False
        self.new_opening = False

    def on_step(self, pose):
        x, y, _ = pose
        if self.st.last_xy is not None:
            self.st.traveled_m += math.hypot(x - self.st.last_xy[0], y - self.st.last_xy[1])
        self.st.last_xy = (x, y)

    def reset_segment(self, pose, step):
        self.st.segment_start = (pose[0], pose[1])

    def mark_called(self, step):
        self.st.last_call_step = step

    def check(self, grid: PlanGrid, pose, step: int, segment_done: bool, path_invalid: bool) -> Optional[str]:
        dc = self.cfg.decision
        openings = count_openings(grid, pose, dc.junction_radius, dc.junction_min_gap_deg)
        new_opening = self.st.prev_openings is not None and openings > self.st.prev_openings
        self.st.prev_openings = openings
        self.new_opening = new_opening
        if path_invalid:
            return "PATH_INVALID"
        if segment_done:
            return "SEGMENT_DONE"
        moved = math.hypot(pose[0] - self.st.segment_start[0], pose[1] - self.st.segment_start[1])
        if new_opening and moved > 0.5:          # 刚出发时地图在补全，缺口数会抖，不算岔口
            return "JUNCTION"
        if (self.expected_path_m and not self._overrun_fired
                and self.st.traveled_m > dc.overrun_factor * self.expected_path_m):
            self._overrun_fired = True
            return "OVERRUN"
        if step - self.st.last_call_step >= dc.llm_fallback_steps:
            return "FALLBACK"
        return None


def subgoal_may_complete(subgoal: Optional[dict], detected_labels: set, n_visible_humans: int) -> bool:
    """当前子目标的完成条件"可能"已满足：它提到的人或物出现在画面里。宁可多问，不可漏问。"""
    if not subgoal:
        return False
    if subgoal.get("ref_human") and n_visible_humans > 0:
        return True
    text = " ".join(str(subgoal.get(k, "")) for k in ("landmark", "done_when")).lower()
    for word, coco in LANDMARK_TO_COCO.items():
        if word in text and coco in detected_labels:
            return True
    return False


def need_llm_call(reason: str, cands: List[Candidate], may_complete: bool, new_opening: bool,
                  final_subgoal: bool = False, skipped_last: bool = False) -> bool:
    """门控：只有"确实没得选"时才跳过，而且不能连续跳过。

    必须调用的情况：
      - PATH_INVALID / OVERRUN / FALLBACK / START：原计划可能有问题
      - 没有任何未走过的可行方向：死胡同或全走过了，回头还是停下是语义判断
      - 已经在最后一个子目标：每个决策点都可能是终点，跳过就会走过头
      - 上一个决策点已经跳过：最多连续跳一次，保证两次询问之间最多隔两段路（约 6 m）
    """
    if reason in ("PATH_INVALID", "OVERRUN", "FALLBACK", "START", "FLOOR", "OBSERVE"):
        return True
    viable = [c for c in cands if c.path is not None and not c.visited]
    if not viable or final_subgoal or skipped_last:
        return True
    return not (len(viable) == 1 and not new_opening and not may_complete)
