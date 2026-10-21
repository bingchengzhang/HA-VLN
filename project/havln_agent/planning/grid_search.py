"""在代价图上找路。

背景：决策点上要给每个候选方向都算一条路线，执行时还要隔几步重算。
目的：一次计算得到"从 agent 出发到窗口内每个格子"的最短路线，候选再多也只算一遍。
本质：代价图 2×2 合并成 0.1 m 的规划栅格（禁区只要有一格是禁区就算禁区，偏保守），
      8 邻接建图，边权 = 几何长度 ×（1 + 平均附加代价 / scale），用 scipy 的 Dijkstra 一次求完。
      这和对每个目标跑 A* 结果一样，但候选多时快得多。
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..mapping.costmap import Costmap

COST_SCALE = 5.0   # 附加代价 20 时，走这一格相当于多走 4 倍路


@dataclass
class PlanGrid:
    lethal: np.ndarray
    wall_lethal: np.ndarray
    cost: np.ndarray
    unknown: np.ndarray
    human_dist: np.ndarray
    dist_wall: np.ndarray       # 到墙距离（池化取最小，偏保守）
    human_lethal: np.ndarray
    factor: int
    cm: Costmap

    @property
    def res(self):
        return self.cm.res * self.factor

    def to_cell(self, x, y):
        i, j = self.cm.world_to_cell(x, y)
        return np.asarray(i) // self.factor, np.asarray(j) // self.factor

    def to_world(self, i, j):
        ci = np.asarray(i) * self.factor + (self.factor - 1) / 2.0
        cj = np.asarray(j) * self.factor + (self.factor - 1) / 2.0
        return self.cm.cell_to_world(ci, cj)

    def inside(self, i, j):
        return (i >= 0) & (j >= 0) & (i < self.lethal.shape[0]) & (j < self.lethal.shape[1])


def _pool(a, f, how):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].reshape(h // f, f, w // f, f)
    return getattr(b, how)(axis=(1, 3))


def make_plan_grid(cm: Costmap, plan_res: float) -> PlanGrid:
    f = max(1, int(round(plan_res / cm.res)))
    return PlanGrid(lethal=_pool(cm.lethal, f, "any"), wall_lethal=_pool(cm.wall_lethal, f, "any"), cost=_pool(cm.cost, f, "max"),
                    unknown=_pool(cm.unknown, f, "all"), human_dist=_pool(cm.human_dist, f, "min"),
                    dist_wall=_pool(cm.dist_wall, f, "min"), human_lethal=_pool(cm.human_lethal, f, "any"),
                    factor=f, cm=cm)


def escape_mask(grid: PlanGrid, x, y) -> np.ndarray:
    """agent 已经落在墙的膨胀禁区里时，允许它走出来的格子。

    背景：禁区是"墙 + 身位 + 余量"，新看到的墙、滑动、地图更新都可能让 agent 站进禁区。
          这时每个方向的第一格都是禁区，候选全空、路线也规划不出，只能原地转圈（服务器上卡了 67 步）。
    本质：只在 agent 周围 wall_r + 一格的圆里放开，而且只放开离墙仍不小于 agent 半径的格
          （不会真的擦墙）、不在人禁区里的格。往墙里走的方向仍然被挡，只有往外的方向能走通。
    agent 不在禁区里时返回全 False，行为与原来完全一样。"""
    i, j = (int(v) for v in grid.to_cell(x, y))
    out = np.zeros_like(grid.wall_lethal)
    if not (grid.inside(i, j) and grid.wall_lethal[i, j]):
        return out
    r = grid.cm.wall_r + grid.res
    k = int(np.ceil(r / grid.res))
    a0, a1 = max(i - k, 0), min(i + k + 1, out.shape[0])
    b0, b1 = max(j - k, 0), min(j + k + 1, out.shape[1])
    ii, jj = np.mgrid[a0:a1, b0:b1]
    disk = np.hypot(ii - i, jj - j) * grid.res <= r
    ok = (grid.dist_wall[a0:a1, b0:b1] >= grid.cm.agent_r) & ~grid.human_lethal[a0:a1, b0:b1]
    out[a0:a1, b0:b1] = disk & ok
    return out


@dataclass
class PathField:
    grid: PlanGrid
    dist: np.ndarray        # (h, w) 从起点出发的加权距离，到不了为 inf
    pred: np.ndarray        # (h*w,) 前驱节点
    start: Tuple[int, int]


def shortest_paths(grid: PlanGrid, start_xy) -> PathField:
    h, w = grid.lethal.shape
    si, sj = (int(v) for v in grid.to_cell(*start_xy))
    si, sj = min(max(si, 0), h - 1), min(max(sj, 0), w - 1)
    free = ~grid.lethal
    # 起点可能因为贴近障碍落在禁区里：放开起点附近一圈，让 agent 能先退出来
    free[max(si - 1, 0):si + 2, max(sj - 1, 0):sj + 2] = True
    free |= escape_mask(grid, *start_xy)

    idx = np.arange(h * w).reshape(h, w)
    rows, cols, wts = [], [], []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = (slice(max(0, -di), h - max(0, di)), slice(max(0, -dj), w - max(0, dj)))
        b = (slice(max(0, di), h - max(0, -di)), slice(max(0, dj), w - max(0, -dj)))
        m = free[a] & free[b]
        if di != 0 and dj != 0:
            # 斜着走不能切角：两个相邻的直角格也必须可通行
            m &= free[(a[0], b[1])] & free[(b[0], a[1])]
        length = grid.res * np.hypot(di, dj)
        wt = length * (1.0 + 0.5 * (grid.cost[a] + grid.cost[b]) / COST_SCALE)
        rows += [idx[a][m], idx[b][m]]
        cols += [idx[b][m], idx[a][m]]
        wts += [wt[m], wt[m]]
    graph = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
    dist, pred = dijkstra(graph, indices=si * w + sj, return_predecessors=True)
    return PathField(grid, dist.reshape(h, w), pred, (si, sj))


def nearest_reachable(field: PathField, x, y, search_m: float = 0.6) -> Optional[Tuple[int, int]]:
    """目标格到不了时（落在禁区里），在附近找一个能到的格。"""
    g = field.grid
    i, j = (int(v) for v in g.to_cell(x, y))
    k = int(np.ceil(search_m / g.res))
    best, best_d = None, np.inf
    for a in range(i - k, i + k + 1):
        for b in range(j - k, j + k + 1):
            if not g.inside(a, b) or not np.isfinite(field.dist[a, b]):
                continue
            d = (a - i) ** 2 + (b - j) ** 2
            if d < best_d:
                best, best_d = (a, b), d
    return best


def extract_path(field: PathField, goal_cell) -> List[Tuple[float, float]]:
    """从前驱表回溯出世界坐标路径（起点 → 终点）。"""
    g = field.grid
    h, w = g.lethal.shape
    node = goal_cell[0] * w + goal_cell[1]
    start = field.start[0] * w + field.start[1]
    cells = []
    while node >= 0 and node != start:
        cells.append(divmod(int(node), w))
        node = field.pred[node]
    cells.append(field.start)
    cells.reverse()
    xs, ys = g.to_world([c[0] for c in cells], [c[1] for c in cells])
    return list(zip(np.atleast_1d(xs).tolist(), np.atleast_1d(ys).tolist()))


def path_length(path) -> float:
    if len(path) < 2:
        return 0.0
    p = np.asarray(path)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())
