"""静态层：2.5D 高程图（一层楼一张）。

背景：固定"离地 0.1–1.6 m 算障碍"会把楼梯当墙（1 m 外的台阶已离地 0.6 m），
      也分不清楼梯、桌面和断崖。
目的：判断每一格"能不能站上去"，和仿真器导航网格的规则一致：台阶不超过 max_step、坡度不太陡。
本质（机器人领域常用的高程图可通行性分析）：
  每格存两个数（世界高度）：
    ground  = 观测到的最低表面高度（只用低于相机的点：地面一定是从上方看到的，相机上方的是天花板）
    top     = ground 以上、身体高度以内的最高点
  每次查询时由这两个数重新判定，看得越全判断越准，不会把一次误判永久记住：
    竖直结构  top − ground > max_step                 墙、家具、桌面下方能看到地面的桌子
    台阶屏障  相邻已知格高差 > max_step + tan(gap_slope)·缝隙宽   桌面边缘、平台边缘、断崖
              缝隙 = 两格之间没观测到的宽度。往下的楼梯从远处看会漏掉几级台阶，按坡度放宽后仍可通行；
              断崖的高差远超坡度允许的范围；水平相机看不到断崖下方，需要低头看一眼（agent 里的 LOOK_DOWN 规则）
  被屏障分隔的两侧，标记较高的那一格为障碍。
栅格坐标：世界坐标 (x, y) 以 episode 起点为原点，格子 (i, j) 中 i 沿 x、j 沿 y。
"""
import math

import numpy as np

from ..config import AgentCfg
from ..geometry import agent_to_world
from .virtual_laser import AgentFramePoints

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


class StaticLayer:
    def __init__(self, cfg: AgentCfg, origin_xy=(0.0, 0.0)):
        m = cfg.map
        self.cfg = cfg
        self.res = m.resolution
        self.n = int(round(m.size_m / m.resolution))
        self.x0 = origin_xy[0] - m.size_m / 2.0
        self.y0 = origin_xy[1] - m.size_m / 2.0
        self.ground = np.full((self.n, self.n), np.inf, dtype=np.float32)    # inf = 未观测
        self.top = np.full((self.n, self.n), -np.inf, dtype=np.float32)
        self.forced = np.zeros((self.n, self.n), dtype=bool)                  # 撞上过的位置
        self._footprint = None
        self.version = 0

    # ---------- 坐标换算 ----------
    def world_to_cell(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        j = np.floor((np.asarray(y) - self.y0) / self.res).astype(np.int64)
        return i, j

    def cell_to_world(self, i, j):
        return self.x0 + (np.asarray(i) + 0.5) * self.res, self.y0 + (np.asarray(j) + 0.5) * self.res

    def in_bounds(self, i, j):
        return (i >= 0) & (j >= 0) & (i < self.n) & (j < self.n)

    # ---------- 更新 ----------
    def update(self, pose, pts: AgentFramePoints, feet_h: float, is_human: np.ndarray = None):
        """pts：本帧反投影的点（agent 坐标，up 相对脚下）；feet_h：agent 脚下的世界高度。"""
        m, s = self.cfg.map, self.cfg.sensor
        sel = ~pts.truncated
        if is_human is not None:
            sel &= ~is_human
        h = feet_h + pts.up
        # 只收本层楼附近的点：楼下太低、楼上太高的不属于这张图
        sel &= (h > feet_h - m.floor_window_below) & (h < feet_h + m.floor_window_above)
        if not sel.any():
            return
        wx, wy = agent_to_world(pose, pts.forward[sel], pts.left[sel])
        i, j = self.world_to_cell(wx, wy)
        hs = h[sel]
        ok = self.in_bounds(i, j)
        i, j, hs = i[ok], j[ok], hs[ok]
        flat = i * self.n + j

        cam_h = feet_h + s.camera_height
        g_cand = hs < cam_h - 0.05
        np.minimum.at(self.ground.reshape(-1), flat[g_cand], hs[g_cand])
        rel = hs - self.ground.reshape(-1)[flat]
        t_sel = (rel > 0.02) & (rel < m.obstacle_max_h)
        np.maximum.at(self.top.reshape(-1), flat[t_sel], hs[t_sel])
        self.version += 1

    def mark_footprint(self, x, y, feet_h):
        """agent 正站着的地方一定是地面：写入脚下高度，清掉那里的"障碍"证据。"""
        r = self.cfg.cost.agent_radius
        k = int(math.ceil(r / self.res))
        ci, cj = self.world_to_cell(x, y)
        ii, jj = np.mgrid[ci - k:ci + k + 1, cj - k:cj + k + 1]
        wx, wy = self.cell_to_world(ii, jj)
        sel = (np.hypot(wx - x, wy - y) <= r) & self.in_bounds(ii, jj)
        self.ground[ii[sel], jj[sel]] = feet_h
        self.top[ii[sel], jj[sel]] = -np.inf
        self.forced[ii[sel], jj[sel]] = False
        self._footprint = (ii[sel], jj[sel])

    def force_obstacle(self, x, y):
        i, j = self.world_to_cell(x, y)
        if self.in_bounds(i, j):
            self.forced[i, j] = True
            self.version += 1

    # ---------- 查询 ----------
    def ground_height(self, x, y) -> float:
        i, j = self.world_to_cell(x, y)
        if not self.in_bounds(i, j):
            return math.nan
        g = float(self.ground[i, j])
        return g if math.isfinite(g) else math.nan

    def classify(self, i0=0, i1=None, j0=0, j1=None) -> np.ndarray:
        """返回窗口内的 int8 栅格：-1 未知，0 可站立，1 障碍。"""
        m = self.cfg.map
        i1 = self.n if i1 is None else i1
        j1 = self.n if j1 is None else j1
        g = self.ground[i0:i1, j0:j1]
        known = np.isfinite(g)
        # 同一阶面的多帧投影会有毫米级差异；容差只用于观测判别，物理跨步上限仍为 max_step_h。
        step_limit = m.max_step_h + m.height_tolerance_m
        vertical = known & (self.top[i0:i1, j0:j1] - g > step_limit)
        barrier = step_barriers(np.where(known, g, np.nan), self.res, step_limit,
                                math.tan(math.radians(m.gap_slope_deg)), int(m.max_gap_m / self.res))
        occ = vertical | barrier | self.forced[i0:i1, j0:j1]
        if self._footprint is not None:
            # 高差分析不能把已经确认由身体占据的区域重新变成墙；只保护本步脚印，外侧断崖仍保留。
            fi, fj = self._footprint
            a, b = fi - i0, fj - j0
            inside = (a >= 0) & (b >= 0) & (a < occ.shape[0]) & (b < occ.shape[1])
            occ[a[inside], b[inside]] = False
        out = np.full(g.shape, UNKNOWN, dtype=np.int8)
        out[known] = FREE
        out[occ] = OCCUPIED
        return out


def step_barriers(g: np.ndarray, res: float, max_step: float, tan_slope: float, max_gap_cells: int) -> np.ndarray:
    """沿 x、y 两个方向扫描：相邻两个已知格（中间可隔着未观测的缝）高差超过允许值，较高的一格记为屏障。
    允许值 = max_step + tan(坡度) × 缝隙宽度。缝隙超过 max_gap_cells 就不比较（离得太远，无从判断）。"""
    occ = np.zeros(g.shape, dtype=bool)
    for G, O in ((g, occ), (g.T, occ.T)):
        n_rows, n_cols = G.shape
        cols = np.arange(n_cols)
        last_h = np.full(n_cols, np.nan)
        last_i = np.full(n_cols, -10 ** 6)
        for i in range(n_rows):
            row = G[i]
            known = np.isfinite(row)
            gap = i - last_i - 1
            cmp = known & np.isfinite(last_h) & (gap <= max_gap_cells)
            if cmp.any():
                diff = row - last_h
                allow = max_step + tan_slope * gap * res
                bad = cmp & (np.abs(diff) > allow)
                higher_here = bad & (diff > 0)
                O[i, higher_here] = True
                lower_here = bad & (diff < 0)
                O[last_i[lower_here], cols[lower_here]] = True
            last_h = np.where(known, row, last_h)
            last_i = np.where(known, i, last_i)
    return occ
