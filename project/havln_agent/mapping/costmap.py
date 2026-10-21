"""代价图：静态层 + 行人层，分别膨胀后合成，只在 agent 周围的窗口里计算。

背景：agent 有体积；按公开源码，被 navmesh 挡住（擦墙或撞人）的那一步就计一次碰撞。
目的：给规划器一张"每格能不能走、走起来多贵"的图。
本质：
  硬禁区 lethal：agent 中心进入即可能被挡（墙按身位 + 余量；人按身位 + 更大余量）
  软代价 cost  ：硬禁区外线性衰减的惩罚，让路线自然离墙、离人远一点（社交距离）
  未知格        ：可以走，但加一点代价（乐观规划，单视角下大量区域是未知的）
整图 48 m 见方有九百多万格，每步整图重算太慢，所以只取 agent 周围 window_m 的窗口。
"""
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage

from ..config import AgentCfg
from .static_layer import StaticLayer, UNKNOWN, OCCUPIED

HumanDisc = Tuple[float, float, float]     # (x, y, 半径)


@dataclass
class Costmap:
    lethal: np.ndarray        # (h, w) bool
    cost: np.ndarray          # (h, w) float，附加代价（不含步长本身）
    human_lethal: np.ndarray  # (h, w) bool，只由人造成的禁区
    wall_lethal: np.ndarray   # (h, w) bool，只由墙造成的禁区
    human_dist: np.ndarray    # (h, w) float，到最近人体（点或圆盘边缘）的距离，风险摘要用
    known_free: np.ndarray    # (h, w) bool
    dist_wall: np.ndarray     # (h, w) float，到最近墙格的距离（未膨胀）
    agent_r: float            # agent 半径：离墙比它近才会真的擦到
    wall_r: float             # 墙禁区半径 = agent 半径 + 墙余量
    unknown: np.ndarray       # (h, w) bool
    i0: int                   # 窗口在静态层里的起始格
    j0: int
    res: float
    layer: StaticLayer

    def world_to_cell(self, x, y):
        i, j = self.layer.world_to_cell(x, y)
        return i - self.i0, j - self.j0

    def cell_to_world(self, i, j):
        return self.layer.cell_to_world(np.asarray(i) + self.i0, np.asarray(j) + self.j0)

    def inside(self, i, j):
        return (i >= 0) & (j >= 0) & (i < self.lethal.shape[0]) & (j < self.lethal.shape[1])

    def is_lethal_world(self, x, y):
        i, j = self.world_to_cell(x, y)
        if not self.inside(i, j):
            return False
        return bool(self.lethal[i, j])


def _ramp(dist, inner, outer, peak):
    span = max(outer - inner, 1e-6)
    return np.clip((outer - dist) / span, 0.0, 1.0) * peak


def build_costmap(layer: StaticLayer, center_xy, human_points: np.ndarray,
                  human_discs: Sequence[HumanDisc], cfg: AgentCfg, human_margin: float = None) -> Costmap:
    """human_margin：临时覆盖人的余量（被人挡太久时放宽），默认用 cfg.cost.human_margin。"""
    c = cfg.cost
    h_margin = c.human_margin if human_margin is None else human_margin
    res = layer.res
    half = int(c.window_m / 2 / res)
    ci, cj = layer.world_to_cell(*center_xy)
    i0, i1 = max(int(ci) - half, 0), min(int(ci) + half, layer.n)
    j0, j1 = max(int(cj) - half, 0), min(int(cj) + half, layer.n)
    grid = layer.classify(i0, i1, j0, j1)
    shape = grid.shape

    # ---- 墙 ----
    occ = grid == OCCUPIED
    dist_wall = ndimage.distance_transform_edt(~occ) * res if occ.any() else np.full(shape, np.inf)
    wall_r = c.agent_radius + c.wall_margin
    wall_lethal = dist_wall < wall_r
    lethal = wall_lethal.copy()
    cost = _ramp(dist_wall, wall_r, wall_r + c.wall_soft_radius, c.soft_cost_max)
    unknown = grid == UNKNOWN
    cost += np.where(unknown, c.unknown_cost, 0.0)

    # ---- 人：看得见的用点（形状精确），看不见的用圆盘 ----
    hum_seed = np.zeros(shape, dtype=bool)
    if human_points is not None and len(human_points):
        hi, hj = layer.world_to_cell(human_points[:, 0], human_points[:, 1])
        hi, hj = hi - i0, hj - j0
        ok = (hi >= 0) & (hj >= 0) & (hi < shape[0]) & (hj < shape[1])
        hum_seed[hi[ok], hj[ok]] = True
    if hum_seed.any():
        dist_h = ndimage.distance_transform_edt(~hum_seed) * res
    else:
        dist_h = np.full(shape, np.inf)
    # 圆盘：到圆盘边缘的距离，与点的距离取最小
    for hx, hy, rad in human_discs:
        di, dj = layer.world_to_cell(hx, hy)
        di, dj = int(di) - i0, int(dj) - j0
        k = int(np.ceil((rad + h_margin + c.human_soft_radius) / res)) + 1
        a0, a1 = max(di - k, 0), min(di + k + 1, shape[0])
        b0, b1 = max(dj - k, 0), min(dj + k + 1, shape[1])
        if a0 >= a1 or b0 >= b1:
            continue
        ii, jj = np.mgrid[a0:a1, b0:b1]
        d = np.hypot((ii - di) * res, (jj - dj) * res) - rad
        dist_h[a0:a1, b0:b1] = np.minimum(dist_h[a0:a1, b0:b1], np.maximum(d, 0.0))

    hum_r = c.agent_radius + h_margin
    human_lethal = dist_h < hum_r
    cost = np.maximum(cost, _ramp(dist_h, hum_r, hum_r + c.human_soft_radius, c.soft_cost_max))
    lethal |= human_lethal

    return Costmap(lethal=lethal, cost=cost, human_lethal=human_lethal, wall_lethal=wall_lethal, human_dist=dist_h,
                   known_free=grid == 0, dist_wall=dist_wall, agent_r=c.agent_radius, wall_r=wall_r,
                   unknown=unknown, i0=i0, j0=j0, res=res, layer=layer)
