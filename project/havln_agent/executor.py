"""执行器：把选定的路线换算成离散动作，并负责紧急覆盖和 STOP 确认。

每步的判断顺序：
  1. 相机还没回正（之前用抬头/低头等待过）→ 先回正，否则建图会把地面投成斜墙
  2. 路线终点已到 → 报告"路段走完"，交给决策层
  3. 朝向没对准前视点 → 转
  4. 前方一步会进入人的禁区，或撞上运动中的人的预测位置 → 原地等（紧急覆盖，大方向不变）
  5. 前方一步会进入墙的禁区 → 报告"路线失效"，交给决策层重算
  6. 前进
官方没有"原地等待"动作，用 LOOK_DOWN / LOOK_UP 成对凑，两步后相机回到原来的俯仰角。
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import AgentCfg
from .geometry import wrap

FORWARD, LEFT, RIGHT, STOP, UP, DOWN = "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "LOOK_UP", "LOOK_DOWN"


@dataclass
class ExecResult:
    action: Optional[str]        # None 表示执行器这一步没有动作，需要决策层介入
    event: Optional[str] = None  # "SEGMENT_DONE" / "PATH_INVALID" / "HUMAN_BLOCKED" / "WAITING"


class Executor:
    def __init__(self, cfg: AgentCfg):
        self.cfg = cfg
        self.path: List[tuple] = []
        self.pitch_steps = 0          # 相对水平的俯仰档数（LOOK_UP +1，LOOK_DOWN −1）
        self.wait_count = 0
        self.wait_phase = 0

    def set_path(self, path, new_target=True):
        """new_target=False 表示只是对同一目标重算路线，等待计数不清零（否则永远等不满，无法升级处理）。"""
        self.path = list(path or [])
        if new_target:
            self.wait_count = 0

    @property
    def pitch(self):
        return math.radians(self.cfg.sensor.tilt_deg) * self.pitch_steps

    def note_action(self, action):
        if action == UP:
            self.pitch_steps += 1
        elif action == DOWN:
            self.pitch_steps -= 1

    def wait_action(self):
        """一对 LOOK_DOWN / LOOK_UP 原地等两步，俯仰角回到原位。"""
        self.wait_count += 1
        self.wait_phase ^= 1
        return DOWN if self.pitch_steps >= 0 else UP

    def level_action(self):
        if self.pitch_steps > 0:
            return DOWN
        if self.pitch_steps < 0:
            return UP
        return None

    def step(self, pose, cm, predicted_humans=(), human_margin=None) -> ExecResult:
        p = self.cfg.plan
        lvl = self.level_action()
        if lvl is not None:
            return ExecResult(lvl)
        if not self.path:
            return ExecResult(None, "SEGMENT_DONE")

        x, y, yaw = pose
        pts = np.asarray(self.path)
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        near = int(np.argmin(d))
        if np.hypot(pts[-1, 0] - x, pts[-1, 1] - y) < 0.3:
            return ExecResult(None, "SEGMENT_DONE")
        # 丢掉已经走过的部分，取前视点
        self.path = self.path[near:]
        pts = pts[near:]
        ahead = np.cumsum(np.r_[0, np.hypot(*np.diff(pts, axis=0).T)]) if len(pts) > 1 else np.zeros(1)
        k = int(np.searchsorted(ahead, p.lookahead_m))
        tx, ty = pts[min(k, len(pts) - 1)]
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        aligned = yaw + round(err / math.radians(self.cfg.sensor.turn_deg)) * math.radians(self.cfg.sensor.turn_deg)
        if not self._can_advance(cm, x, y, aligned, predicted_humans, human_margin):
            alternative = self._corner_heading(pose, cm, pts, ahead, predicted_humans, human_margin)
            if alternative is not None:
                err = wrap(alternative - yaw)
        if abs(err) > math.radians(p.turn_tol_deg) + 1e-6:
            return ExecResult(LEFT if err > 0 else RIGHT)

        step = self.cfg.sensor.step_len
        nx, ny = x + step * math.cos(yaw), y + step * math.sin(yaw)
        hit_human = self._segment_hits(cm, x, y, nx, ny, cm.human_lethal)
        r_h = self.cfg.cost.agent_radius + (self.cfg.cost.human_margin if human_margin is None else human_margin)
        hit_pred = any(math.hypot(nx - hx, ny - hy) < r_h + rad for hx, hy, rad in predicted_humans)
        if hit_human or hit_pred:
            if self.wait_count < p.max_wait_steps or self.wait_phase == 1:
                return ExecResult(self.wait_action(), "WAITING")
            self.wait_count = 0
            return ExecResult(None, "HUMAN_BLOCKED")
        if self._segment_hits(cm, x, y, nx, ny, cm.wall_lethal) and not self._escaping(cm, x, y, nx, ny):
            return ExecResult(None, "PATH_INVALID")
        if self.wait_phase == 1:                        # 等待做了一半：补齐，让相机回正
            return ExecResult(self.wait_action(), "WAITING")
        self.wait_count = 0
        return ExecResult(FORWARD)

    def _can_advance(self, cm, x, y, yaw, predicted_humans, human_margin):
        step = self.cfg.sensor.step_len
        nx, ny = x + step * math.cos(yaw), y + step * math.sin(yaw)
        i, j = cm.world_to_cell(nx, ny)
        if not cm.inside(i, j):
            return False
        radius = self.cfg.cost.agent_radius + (self.cfg.cost.human_margin if human_margin is None else human_margin)
        if self._segment_hits(cm, x, y, nx, ny, cm.human_lethal):
            return False
        if any(math.hypot(nx - hx, ny - hy) < radius + rad for hx, hy, rad in predicted_humans):
            return False
        return not self._segment_hits(cm, x, y, nx, ny, cm.wall_lethal) or self._escaping(cm, x, y, nx, ny)

    def _corner_heading(self, pose, cm, points, distances, predicted_humans, human_margin):
        """前视直线切过弯角时，选择沿原路径有进展且整步安全的离散朝向。"""
        x, y, yaw = pose
        step = self.cfg.sensor.step_len
        limit = max(self.cfg.plan.lookahead_m, 4 * step)
        count = min(int(np.searchsorted(distances, limit)) + 1, len(points))
        points, distances = points[:count], distances[:count]
        turn = math.radians(self.cfg.sensor.turn_deg)
        best, best_score = None, -math.inf
        for k in range(int(round(2 * math.pi / turn))):
            heading = yaw + k * turn
            if not self._can_advance(cm, x, y, heading, predicted_humans, human_margin):
                continue
            nx, ny = x + step * math.cos(heading), y + step * math.sin(heading)
            errors = np.hypot(points[:, 0] - nx, points[:, 1] - ny)
            nearest = int(np.argmin(errors))
            progress, error = distances[nearest], errors[nearest]
            if progress < .5 * step or error > .15:
                continue
            score = progress - 2 * error - .02 * abs(wrap(heading - yaw))
            if score > best_score:
                best, best_score = heading, score
        return best

    @staticmethod
    def _escaping(cm, x0, y0, x1, y1) -> bool:
        """已经站在墙禁区里：这一步终点离墙比起点远，且沿途离墙始终不小于 agent 半径，就算在往外走。"""
        i0, j0 = cm.world_to_cell(x0, y0)
        if not (cm.inside(i0, j0) and cm.wall_lethal[i0, j0]):
            return False
        d0 = cm.dist_wall[i0, j0]
        ds = []
        for s in np.linspace(0.3, 1.0, 4):
            i, j = cm.world_to_cell(x0 + s * (x1 - x0), y0 + s * (y1 - y0))
            if not cm.inside(i, j):
                return False
            ds.append(cm.dist_wall[i, j])
        return min(ds) >= cm.agent_r and ds[-1] > d0

    @staticmethod
    def _segment_hits(cm, x0, y0, x1, y1, mask) -> bool:
        for s in np.linspace(0.3, 1.0, 4):           # 起点附近不查：agent 可能正贴着禁区边缘
            i, j = cm.world_to_cell(x0 + s * (x1 - x0), y0 + s * (y1 - y0))
            if cm.inside(i, j) and mask[i, j]:
                return True
        return False


def goal_depth(depth_m: np.ndarray, side: str) -> float:
    """STOP 确认的退路（模型没给 goal_box 时）：画面对应三分之一、中间高度带的深度中位数。比较粗。"""
    h, w = depth_m.shape
    cols = {"left": slice(0, w // 3), "center": slice(w // 3, 2 * w // 3), "right": slice(2 * w // 3, w)}[side]
    band = depth_m[int(h * 0.35):int(h * 0.75), cols]
    band = band[band > 0.05]
    return float(np.median(band)) if band.size else math.inf


def box_depth_bearing(depth_m: np.ndarray, box, box_side: int, hfov_deg: float):
    """goal_box（box_side 尺度的像素框）→（目标深度，目标方位角）。
    深度取框内中间一半区域的中位数，避开框边缘混进来的背景；方位角取框中心列。"""
    h, w = depth_m.shape
    sx, sy = w / box_side, h / box_side
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max((x2 - x1) / 4, 1), max((y2 - y1) / 4, 1)
    c0, c1 = int((cx - hw) * sx), int(np.ceil((cx + hw) * sx))
    r0, r1 = int((cy - hh) * sy), int(np.ceil((cy + hh) * sy))
    patch = depth_m[max(r0, 0):min(r1, h), max(c0, 0):min(c1, w)]
    patch = patch[patch > 0.05]
    d = float(np.median(patch)) if patch.size else math.inf
    focal = (box_side / 2) / math.tan(math.radians(hfov_deg) / 2)
    bearing = math.atan((box_side / 2 - cx) / focal)          # 向左为正
    return d, bearing


def side_bearing(side: str, hfov_deg: float) -> float:
    """画面左/中/右三分之一的中心对应的方位角（向左为正，弧度）。"""
    return {"left": 1, "center": 0, "right": -1}[side] * math.radians(hfov_deg) / 3.0
