"""行人层：每帧看到的人体点 + 跨帧跟踪的行人记忆。

背景：人和墙必须分开处理。撞人的代价更大；人在开局 12 秒内会动；
      单视角只有 90°，转身后人就出视野，但他还在那里。
目的：给代价图两样东西——此刻看得见的人体占据（精确的形状），
      以及看不见的人的圆盘（位置 + 随时间增大的不确定半径）。
本质：
  1. 每个检测实例的掩码像素反投影成世界坐标点 → 本帧人体占据
  2. 每个实例取点的质心，沿视线往后补半个身体厚度 → 人的中心
  3. 中心跨帧做最近邻关联，估速度；看不见的人保留，不确定半径随时间增长
避障用点（形状不丢），跟踪用中心（有身份），这两件事刻意分开。
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..config import AgentCfg
from ..geometry import agent_to_world
from .virtual_laser import AgentFramePoints


@dataclass
class Track:
    tid: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    last_seen: float = 0.0       # 秒（episode 内的时间）
    hits: int = 1
    z: float = 0.0               # 人体点的中位世界高度（约在腰部），用于区分楼层

    def speed(self):
        return float(np.hypot(self.vx, self.vy))


@dataclass
class HumanObservation:
    """一帧的行人观测（世界坐标）。"""
    points_xy: np.ndarray                           # (N, 2) 本帧所有人体点（障碍高度带内）
    centers: List[tuple] = field(default_factory=list)   # 每个实例的中心 (x, y, z)


def observe_humans(pts: AgentFramePoints, instance_ids: np.ndarray, pose, cfg: AgentCfg,
                   feet_h: float = 0.0) -> HumanObservation:
    """instance_ids: 与深度图同尺寸的 int 图，0 为背景，k>0 为第 k 个人。"""
    m, h = cfg.map, cfg.human
    if instance_ids is None:
        return HumanObservation(np.zeros((0, 2)))
    inst = instance_ids[pts.rows, pts.cols]
    band = (pts.up > m.obstacle_min_h) & (pts.up < m.obstacle_max_h) & ~pts.truncated
    sel = (inst > 0) & band
    wx, wy = agent_to_world(pose, pts.forward[sel], pts.left[sel])
    points = np.stack([wx, wy], axis=1) if sel.any() else np.zeros((0, 2))

    centers = []
    for k in np.unique(inst[sel]):
        mk = inst[sel] == k
        if mk.sum() < h.min_mask_pixels // 4:       # 行方向已隔行采样，阈值相应缩小
            continue
        # 用中位数而不是均值：掩码边缘偶尔混入背景像素
        f = np.median(pts.forward[sel][mk])
        l = np.median(pts.left[sel][mk])
        r = np.hypot(f, l)
        scale = (r + h.body_half_depth) / max(r, 1e-6)   # 沿视线往后补半个身体厚度
        cx, cy = agent_to_world(pose, f * scale, l * scale)
        cz = feet_h + float(np.median(pts.up[sel][mk]))
        centers.append((float(cx), float(cy), cz))
    return HumanObservation(points, centers)


class HumanLayer:
    def __init__(self, cfg: AgentCfg):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.visible_points = np.zeros((0, 2))
        self._next_id = 1

    def update(self, obs: HumanObservation, t: float, visible_fn=None):
        """visible_fn(x, y) -> bool：这个位置此刻是否在视野内且没被遮挡。
        在视野内却没检测到的轨迹，说明人已经走开了，直接删掉，避免留下"幽灵"。"""
        h = self.cfg.human
        self.visible_points = obs.points_xy
        unmatched = list(range(len(self.tracks)))
        for cx, cy, cz in obs.centers:
            best, best_d = None, h.assoc_dist
            for idx in unmatched:
                tr = self.tracks[idx]
                if abs(tr.z - cz) > 1.0:
                    continue
                px, py = self._predict(tr, t)
                d = float(np.hypot(px - cx, py - cy))
                if d < best_d:
                    best, best_d = idx, d
            if best is None:
                self.tracks.append(Track(self._next_id, cx, cy, last_seen=t, z=cz))
                self._next_id += 1
                continue
            unmatched.remove(best)
            tr = self.tracks[best]
            dt = max(t - tr.last_seen, 1e-3)
            vx, vy = (cx - tr.x) / dt, (cy - tr.y) / dt
            a = h.vel_smooth
            tr.vx, tr.vy = (1 - a) * tr.vx + a * vx, (1 - a) * tr.vy + a * vy
            tr.x, tr.y, tr.z, tr.last_seen, tr.hits = cx, cy, cz, t, tr.hits + 1

        keep = []
        for idx, tr in enumerate(self.tracks):
            if t - tr.last_seen > h.track_ttl_s:
                continue
            if idx in unmatched and visible_fn is not None and visible_fn(tr.x, tr.y):
                continue                                   # 该看见却没看见：人已离开
            keep.append(tr)
        self.tracks = keep

    def _same_floor(self, tr: Track, feet_h):
        """人体中位点大约在脚上方 0.9 m；相差超过 1.5 m 就不在同一层。"""
        return feet_h is None or abs(tr.z - (feet_h + 0.9)) < 1.5

    def _predict(self, tr: Track, t: float):
        h = self.cfg.human
        # 行人只在开局 motion_duration_s 内运动；之后速度归零，不再外推
        dt = max(0.0, min(t, h.motion_duration_s) - min(tr.last_seen, h.motion_duration_s))
        dt = min(dt, h.predict_horizon_s * 3)
        if tr.speed() < h.static_speed:
            return tr.x, tr.y
        return tr.x + tr.vx * dt, tr.y + tr.vy * dt

    def unseen_discs(self, t: float, agent_xy=None, feet_h=None):
        """此刻看不见的人：(x, y, 半径)。看得见的人由 visible_points 精确表示。

        不确定半径只给"在动或还没看清动没动"的人加：被看到过多次且静止的人，看不见时也还在原处。
        圆盘不能把 agent 自己罩进去：agent 就站在那里而没有被挡，说明人不在那么近的地方。
        """
        h, c = self.cfg.human, self.cfg.cost
        out = []
        for tr in self.tracks:
            age = t - tr.last_seen
            if age <= 1e-6 or not self._same_floor(tr, feet_h):
                continue
            settled = tr.hits >= 3 and tr.speed() < h.static_speed
            extra = 0.0
            if not settled:
                moving_time = max(0.0, min(t, h.motion_duration_s) - min(tr.last_seen, h.motion_duration_s))
                extra = min(h.uncert_growth * moving_time, h.uncert_max)
            px, py = self._predict(tr, t)
            rad = c.unseen_body_radius + extra
            if agent_xy is not None:
                dist = float(np.hypot(px - agent_xy[0], py - agent_xy[1]))
                rad = min(rad, max(0.5 * c.unseen_body_radius, dist - c.agent_radius - c.human_margin_relaxed - 0.05))
            out.append((px, py, rad))
        return out

    def predicted_discs(self, t: float, horizon: float, feet_h=None):
        """正在走动的人在 horizon 秒后的位置（只在开局运动期内有意义），给执行器避开未来位置。
        静止的人不在这里：他们的形状已经由看得见的点或看不见的圆盘表示了，再加一遍会把路堵死。"""
        c, h = self.cfg.cost, self.cfg.human
        out = []
        for tr in self.tracks:
            if tr.speed() < h.static_speed or not self._same_floor(tr, feet_h):
                continue
            px, py = self._predict(tr, t + horizon)
            out.append((px, py, c.unseen_body_radius))
        return out

    def describe(self, pose, feet_h=None):
        """给慢回路的结构化描述：每个已知行人相对 agent 的距离、方位、是否静止。"""
        from ..geometry import world_to_agent
        out = []
        for tr in self.tracks:
            if not self._same_floor(tr, feet_h):
                continue
            f, l = world_to_agent(pose, tr.x, tr.y)
            dist = float(np.hypot(f, l))
            ang = float(np.degrees(np.arctan2(l, f)))
            side = "前方" if abs(ang) < 30 else ("左前" if 0 < ang < 90 else "右前" if -90 < ang < 0 else "后方")
            out.append({"id": tr.tid, "distance_m": round(dist, 1), "bearing_deg": round(ang),
                        "side": side, "static": tr.speed() < self.cfg.human.static_speed})
        return out
