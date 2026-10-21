"""特权专家：用参考路径和真实终点给决策点打标签。

只用于三件事，绝不进入正式推理：
  1. 生成慢回路训练数据（scripts/gen_decision_data.py）
  2. 逐层替换诊断：--oracle-decisions 让专家代替大模型，测"候选机制 + 建图"的上限
  3. 事后诊断指标：候选覆盖率、选错率、错误停止、错误楼层

标签规则：
  choose   参考路径上"当前进度往前 3 m"的点，选终点离它最近的候选
  stop     离终点 2 m 内、同一楼层、终点在视野里
  goal_box 终点投影到当前画面的位置（以终点地面为中心、人高 1.6 m 的竖条）
  observe  最近的候选离目标点也超过 1.5 m（正确方向不在候选里），且目标点不在视野内：转过去看
  subgoal_done 按参考路径完成比例 × 子目标数估计，是弱标签
"""
import math
from typing import List, Optional, Tuple

import numpy as np

from .config import AgentCfg
from .geometry import bearing_to, world_to_agent

AHEAD_M = 3.0
STOP_R = 2.0
COVER_R = 1.5


def progress_target(ref: List[Tuple[float, float, float]], x, y, h, ahead_m=AHEAD_M):
    """ref：参考路径 [(x, y, 高度)]。返回（目标点 (x, y, h)，已完成比例）。
    投影到参考线段后沿线插值，避免稀疏路点把“前方3米”变成十几米。
    同高度优先，避免楼上楼下同一位置混淆。"""
    p = np.asarray(ref, dtype=float)
    if len(p) == 0:
        raise ValueError("参考路径为空")
    if len(p) == 1:
        return tuple(map(float, p[0])), 1.0
    delta = np.diff(p, axis=0)
    seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    squared = np.sum(delta[:, :2] ** 2, axis=1)
    alpha = np.sum((np.array([x, y]) - p[:-1, :2]) * delta[:, :2], axis=1) / np.maximum(squared, 1e-12)
    vertical = squared < 1e-12
    rising = vertical & (np.abs(delta[:, 2]) > 1e-12)
    alpha[rising] = (h - p[:-1, 2][rising]) / delta[:, 2][rising]
    alpha = np.clip(alpha, 0., 1.)
    projected = p[:-1] + alpha[:, None] * delta
    d = np.hypot(projected[:, 0] - x, projected[:, 1] - y) + \
        10.0 * (np.abs(projected[:, 2] - h) > 1.0)
    k = int(np.argmin(d))
    progress = cum[k] + alpha[k] * seg[k]
    distance = min(progress + ahead_m, cum[-1])
    if distance >= cum[-1]:
        target = p[-1]
    else:
        j = min(int(np.searchsorted(cum, distance, side='right')) - 1, len(seg) - 1)
        fraction = (distance - cum[j]) / max(seg[j], 1e-12)
        target = p[j] + fraction * delta[j]
    return tuple(map(float, target)), float(progress / max(cum[-1], 1e-6))


def project_to_image(cfg: AgentCfg, pose, feet_h, wx, wy, wz, side: int):
    """世界点 → 当前画面（边长 side 的正方形图）里的像素坐标；在相机后面返回 None。"""
    f, l = world_to_agent(pose, wx, wy)
    if f <= 0.1:
        return None
    up = wz - (feet_h + cfg.sensor.camera_height)
    focal = (side / 2.0) / math.tan(math.radians(cfg.sensor.hfov_deg) / 2.0)
    u = side / 2.0 - focal * (l / f)
    v = side / 2.0 - focal * (up / f)
    return u, v


def goal_box(cfg: AgentCfg, pose, feet_h, goal, side: int) -> Optional[list]:
    """终点在画面中的框：终点地面到其上 1.6 m 的竖条，宽约 0.6 m。完全出画面返回 None。"""
    gx, gy, gz = goal
    lo = project_to_image(cfg, pose, feet_h, gx, gy, gz, side)
    hi = project_to_image(cfg, pose, feet_h, gx, gy, gz + 1.6, side)
    if lo is None or hi is None:
        return None
    f, _ = world_to_agent(pose, gx, gy)
    focal = (side / 2.0) / math.tan(math.radians(cfg.sensor.hfov_deg) / 2.0)
    half_w = max(6.0, focal * 0.3 / f)
    x1, x2 = lo[0] - half_w, lo[0] + half_w
    y1, y2 = hi[1], lo[1]
    if x2 < 0 or x1 > side or y2 < 0 or y1 > side:
        return None
    clip = lambda v: int(round(min(max(v, 0), side - 1)))
    return [clip(x1), clip(y1), clip(x2), clip(y2)]


class OracleDecider:
    """按参考路径做决策。label() 返回专家标签（slow_loop.Decision）。"""

    def __init__(self, cfg: AgentCfg, ref_path, goal):
        self.cfg, self.ref, self.goal = cfg, ref_path, goal

    def label(self, agent, cands, pose):
        from .slow.slow_loop import Decision
        c = self.cfg
        feet = agent.feet
        gx, gy, gz = self.goal
        tgt, frac = progress_target(self.ref, pose[0], pose[1], feet)
        n_sub = len(agent.plan["subgoals"])
        sub_done = int(frac * n_sub) > agent.sub_idx and agent.sub_idx < n_sub - 1

        half = c.sensor.hfov_deg / 2
        d_goal = math.dist((pose[0], pose[1], feet), (gx, gy, gz))
        b_goal = math.degrees(bearing_to(pose, gx, gy))
        if d_goal < STOP_R and abs(gz - feet) < 1.0 and abs(b_goal) < half:
            box = goal_box(c, pose, feet, self.goal, c.slow.image_side)
            side = "left" if b_goal > half / 3 else ("right" if b_goal < -half / 3 else "center")
            return Decision(None, sub_done, True, side, "expert: goal reached", "expert", goal_box=box)

        viable = [k for k in cands if k.path is not None]
        # 路被人暂时挡住时仍选择该候选，由执行器等待/重规划；否则专家只会原地观察，永远触发不了等待升级。
        options = viable or [k for k in cands if getattr(k, 'blocked_by_human', False)]
        best = min(options, key=lambda k: math.hypot(k.endpoint[0] - tgt[0], k.endpoint[1] - tgt[1]), default=None)
        best_d = math.inf if best is None else math.hypot(best.endpoint[0] - tgt[0], best.endpoint[1] - tgt[1])
        b_tgt = math.degrees(bearing_to(pose, tgt[0], tgt[1]))
        if (best_d > COVER_R and abs(b_tgt) > half and not agent.observed_last
                and agent.n_observes < c.decision.max_observes):
            look = "behind" if abs(b_tgt) > 135 else ("left" if b_tgt > 0 else "right")
            return Decision(None, sub_done, False, "none", "expert: look", "expert", observe=look)
        return Decision(None if best is None else best.cid, sub_done, False, "none", "expert", "expert")

    def covered(self, cands, pose, feet) -> bool:
        """候选覆盖：有没有一个候选的终点离"进度目标点"不到 COVER_R。"""
        tgt, _ = progress_target(self.ref, pose[0], pose[1], feet)
        return any(math.hypot(k.endpoint[0] - tgt[0], k.endpoint[1] - tgt[1]) < COVER_R for k in cands)
