"""深度图 → 反投影点云，以及一条用于"可见性判断"的虚拟激光。

建图用的是反投影点（见 static_layer.py 的高程图）；虚拟激光现在只用来判断
"某个位置此刻是否在视野内且没被挡住"（行人层据此删除已离开的人），所以仍按固定高度带取最近点。

背景：规划只需要一张 2D 俯视图，把 6 万多个深度像素逐个投进栅格既慢又没必要。
目的：把一帧深度压成 N 条水平射线，每条给出"这个方向多远处有障碍"和"至少多远是空的"。
本质：先把像素反投影到 agent 坐标系（考虑俯仰角），再按水平方位角分桶：
      桶内障碍高度带里最近的点 = 射线终点（命中）；
      桶内没有障碍时，看到的最远地面点 = 射线长度（不命中，只证明这段是空的）。

坐标约定（agent 坐标系，单位米）：
  forward  水平向前
  left     水平向左
  up       向上，地面为 0
方位角 bearing 以正前方为 0，向左为正（逆时针）。
"""
from dataclasses import dataclass

import numpy as np

from ..config import AgentCfg


@dataclass
class Scan:
    bearings: np.ndarray   # (N,) 每条射线相对正前方的方位角（弧度）
    ranges: np.ndarray     # (N,) 射线长度（米）
    is_hit: np.ndarray     # (N,) 终点是否是障碍；False 表示只知道这段是空的
    valid: np.ndarray      # (N,) 这条射线有没有任何信息


@dataclass
class AgentFramePoints:
    """一帧深度反投影后的点（行方向已下采样，均为一维数组）。"""
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray            # 离地高度
    truncated: np.ndarray     # 该像素深度达到截断值（"这个方向 max_depth 内没东西"）
    rows: np.ndarray          # 对应原图的行列号，便于和人体掩码对齐
    cols: np.ndarray


def denormalize_depth(depth: np.ndarray, cfg: AgentCfg) -> np.ndarray:
    """Habitat 的深度可能归一化到 [0,1]，还原成米；去掉末尾的通道维。"""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if cfg.sensor.depth_normalized:
        d = d * cfg.sensor.max_depth
    return d


def backproject(depth_m: np.ndarray, cfg: AgentCfg, pitch: float, row_stride: int = 2) -> AgentFramePoints:
    """深度图（米，Habitat 的 z 深度，即沿光轴的距离）→ agent 坐标系下的点。

    pitch: 相机俯仰角，抬头为正（弧度）。
    相机系：x 向右、y 向下、z 向前。抬头 pitch 后：
      水平前向 = z·cos(p) + y·sin(p)
      离地高度 = cam_h + z·sin(p) − y·cos(p)
    """
    h, w = depth_m.shape
    # 列不下采样：图像列在角度上不均匀（中央最稀），隔列取会让中央的方位桶落空
    rows, cols = np.mgrid[0:h:row_stride, 0:w]
    rows, cols = rows.ravel(), cols.ravel()
    z = depth_m[rows, cols]

    f = cfg.focal_px * (w / cfg.sensor.img_width)       # 图像被缩放时焦距按比例缩放
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x_right = (cols - cx) / f * z
    y_down = (rows - cy) / f * z

    cp, sp = np.cos(pitch), np.sin(pitch)
    forward = z * cp + y_down * sp
    up = cfg.sensor.camera_height + z * sp - y_down * cp
    truncated = z >= cfg.sensor.max_depth * 0.99
    keep = z > 0.05                                      # 0 深度是无效像素（贴脸或空洞）
    return AgentFramePoints(forward[keep], -x_right[keep], up[keep], truncated[keep], rows[keep], cols[keep])


def points_to_scan(pts: AgentFramePoints, cfg: AgentCfg, human_mask: np.ndarray = None) -> Scan:
    """按方位角分桶，得到虚拟激光。

    human_mask: 与深度图同尺寸的布尔图，True 为人。人的点会挡住射线（射线到此为止），
    但不算静态障碍——人交给行人层处理。
    """
    m = cfg.map
    n = m.n_virtual_rays
    half = np.radians(cfg.sensor.hfov_deg) / 2.0
    bin_edges_width = 2 * half / n

    horiz = np.hypot(pts.forward, pts.left)
    bearing = np.arctan2(pts.left, pts.forward)
    idx = np.floor((bearing + half) / bin_edges_width).astype(np.int64)
    in_fov = (idx >= 0) & (idx < n) & (pts.forward > 0)

    is_human = np.zeros_like(horiz, dtype=bool)
    if human_mask is not None:
        is_human = human_mask[pts.rows, pts.cols]

    in_band = (pts.up > m.obstacle_min_h) & (pts.up < m.obstacle_max_h)
    obstacle = in_fov & in_band & ~is_human & ~pts.truncated
    blocker_h = in_fov & is_human & ~pts.truncated
    # 空地证据：看到的地面点；以及在高度带内却达到截断深度的点（这个方向一直空到 max_depth）
    floor = in_fov & (np.abs(pts.up) < m.max_step_h) & ~is_human
    open_far = in_fov & in_band & pts.truncated & ~is_human

    inf = np.inf
    obs_min = np.full(n, inf)
    hum_min = np.full(n, inf)
    free_max = np.zeros(n)
    np.minimum.at(obs_min, idx[obstacle], horiz[obstacle])
    np.minimum.at(hum_min, idx[blocker_h], horiz[blocker_h])
    ev = floor | open_far
    np.maximum.at(free_max, idx[ev], horiz[ev])

    blocker = np.minimum(obs_min, hum_min)
    has_blocker = np.isfinite(blocker)
    ranges = np.where(has_blocker, blocker, free_max)
    is_hit = has_blocker & (obs_min <= hum_min)
    valid = has_blocker | (free_max > 0)

    bearings = -half + (np.arange(n) + 0.5) * bin_edges_width
    return Scan(bearings, ranges, is_hit, valid)


def depth_to_scan(depth, cfg: AgentCfg, pitch: float = 0.0, human_mask=None, row_stride: int = 2) -> Scan:
    """便捷入口：原始深度观测 → 虚拟激光。"""
    pts = backproject(denormalize_depth(depth, cfg), cfg, pitch, row_stride)
    if human_mask is not None:
        human_mask = np.asarray(human_mask, dtype=bool)
    return points_to_scan(pts, cfg, human_mask)
