"""候选方向 + 风险摘要。

背景：大模型不擅长从画面里读出"哪条路能走、哪条路有人"，AHead-VLN 的消融说明
      去掉结构化风险输入后成功率从 0.17 掉到 0.06。
目的：在决策点把"往哪走"变成几道带说明的选择题。
本质：
  1. 以 agent 为中心每 15° 打一条射线，只看墙：能直行多远
  2. 连续可行的方向合并成一组，太宽的组按 candidate_merge_deg 拆开，每组取能走最远的方向
  3. 终点用同一次 Dijkstra 的结果回溯路线，再沿路线统计风险：
     多远、离人最近多少、是否穿过人的社交区、是否通向未知、是否走过
  4. 墙上能走但被人挡住的方向也保留（标 blocked_by_human），是否等人让开由大模型结合指令决定
"""
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from ..config import AgentCfg
from ..geometry import wrap
from .grid_search import PlanGrid, PathField, extract_path, nearest_reachable, path_length, escape_mask


@dataclass
class Candidate:
    cid: int
    bearing_deg: float               # 相对当前朝向，向左为正
    abs_yaw: float                   # 地图绝对朝向
    endpoint: tuple                  # 终点 (x, y)
    straight_m: float                # 只看墙时能直行多远
    path: Optional[list] = None      # 规划路线；None 表示目前到不了
    distance_m: float = 0.0
    min_clearance_to_human_m: float = 99.0
    crosses_human_zone: bool = False
    blocked_by_human: bool = False
    leads_to_unknown: bool = False
    visited: bool = False
    elevation_change_m: Optional[float] = None   # 路线上最远的已知地面比脚下高多少；None 表示一路都没看到地面
    tried_before: Optional[str] = None           # 以前这个方向失败过的原因："blocked_by_human" / "dead_end"
    in_view: bool = False
    view_ref: Optional[int] = None   # 视野记忆里对应的图片编号

    def summary(self):
        """送给慢回路的字段（不含路线坐标）。"""
        d = asdict(self)
        for k in ("path", "abs_yaw", "endpoint", "view_ref"):
            d.pop(k)
        for k, v in d.items():                      # numpy 的 bool/float 不能直接 json.dumps
            if isinstance(v, np.generic):
                d[k] = v.item()
        d["bearing_deg"] = int(round(self.bearing_deg))
        for k in ("straight_m", "distance_m", "min_clearance_to_human_m"):
            d[k] = round(float(d[k]), 1)
        e = self.elevation_change_m
        d["elevation_change_m"] = None if e is None else round(float(e), 1)
        d["goes"] = "unknown" if e is None else ("up" if e > 0.5 else "down" if e < -0.5 else "level")
        d["direction"] = describe_bearing(self.bearing_deg)
        return d


def describe_bearing(b):
    a = abs(b)
    if a <= 15:
        return "正前方"
    side = "左" if b > 0 else "右"
    if a <= 60:
        return side + "前方"
    if a <= 120:
        return side + "侧"
    if a < 165:
        return side + "后方"
    return "正后方"


def _ray_free(grid: PlanGrid, x, y, yaw, max_d, blocked: np.ndarray):
    """沿 yaw 方向逐格走，返回碰到 blocked 之前的距离。"""
    step = grid.res * 0.5
    d = step
    while d <= max_d:
        i, j = grid.to_cell(x + d * math.cos(yaw), y + d * math.sin(yaw))
        if not grid.inside(i, j) or blocked[i, j]:
            return d - step
        d += step
    return max_d


def generate_candidates(grid: PlanGrid, field: PathField, pose, visited_xy: np.ndarray,
                        cfg: AgentCfg, exclude_yaws=(), ground_fn=None, feet_h: float = 0.0) -> List[Candidate]:
    """ground_fn(x, y) -> 该点地面的世界高度（未观测为 nan），用来标出上楼/下楼的候选。"""
    dc, cc = cfg.decision, cfg.cost
    x, y, yaw = pose
    n = dc.n_candidate_dirs
    step_deg = 360.0 / n
    rel = np.array([wrap(math.radians(k * step_deg)) for k in range(n)])
    wall_only = grid.wall_lethal & ~escape_mask(grid, x, y)
    reach_wall = np.array([_ray_free(grid, x, y, yaw + r, dc.candidate_max_dist, wall_only) for r in rel])
    feasible = reach_wall >= dc.candidate_min_dist
    short_candidates = not feasible.any()
    if short_candidates:
        # 狭窄拐角未必能直行一米。仅在正常候选全空时使用可执行短段；不放宽墙/人体禁区。
        short_min = max(2 * cfg.sensor.step_len, cfg.plan.lookahead_m)
        feasible = reach_wall >= short_min

    # 环形连续段：找到一个不可行的方向作为起点，把可行方向切成若干段
    groups = []
    if feasible.all():
        start = 0
    elif not feasible.any():
        start = 0  # 直线全空时仍检查最短路径场是否存在需要拐弯的出口。
    else:
        start = int(np.argmin(feasible))
    cur = []
    for s in range(n):
        k = (start + s) % n
        if feasible[k]:
            cur.append(k)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    max_per = max(1, int(round(dc.candidate_merge_deg / step_deg)))
    picks = []
    for g in groups:
        for c0 in range(0, len(g), max_per):
            chunk = g[c0:c0 + max_per]
            best = max(chunk, key=lambda k: (reach_wall[k], -abs(rel[k])))
            picks.append(best)

    half_fov = math.radians(cfg.sensor.hfov_deg) / 2
    cands = []
    for k in picks:
        a = yaw + rel[k]
        if any(abs(wrap(a - e)) < math.radians(step_deg) for e in exclude_yaws):
            continue
        d = min(reach_wall[k], dc.candidate_max_dist)
        ex, ey = x + d * math.cos(a), y + d * math.sin(a)
        cand = Candidate(cid=0, bearing_deg=math.degrees(rel[k]), abs_yaw=wrap(a), endpoint=(ex, ey),
                         straight_m=float(reach_wall[k]), in_view=abs(rel[k]) <= half_fov)
        goal = nearest_reachable(field, ex, ey)
        if goal is None:
            cand.blocked_by_human = True          # 墙上可行但规划不到：被人（或人的余量）挡住
        else:
            cand.path = extract_path(field, goal)
            _annotate(cand, grid, cfg)
            if ground_fn is not None:
                # 终点常落在还没看到的地方：取路线上最远一个已观测到地面的点
                for px, py in reversed(cand.path):
                    gh = ground_fn(px, py)
                    if not math.isnan(gh):
                        cand.elevation_change_m = gh - feet_h
                        break
        if short_candidates and cand.path is not None and \
                math.hypot(cand.path[-1][0] - x, cand.path[-1][1] - y) < 0.35:
            cand.path = None
            cand.blocked_by_human = True  # 近端吸附并非路线恢复，保留等待语义而不是伪造已到达的短段。
        if len(visited_xy):
            cand.visited = bool(np.min(np.hypot(visited_xy[:, 0] - ex, visited_xy[:, 1] - ey)) < 0.75)
        cands.append(cand)

    viable = [c for c in cands if c.path is not None]
    if short_candidates or not viable or all(c.visited for c in viable):
        cands += _curved_candidates(grid, field, pose, visited_xy, cfg, ground_fn, feet_h, exclude_yaws)
    cands = _merge_same_way(cands)
    # 编号按方位从左到右，方便大模型和图片对应
    cands.sort(key=lambda c: -c.bearing_deg)
    for i, c in enumerate(cands):
        c.cid = i
    return cands


def _curved_candidates(grid, field, pose, visited_xy, cfg, ground_fn, feet_h, exclude_yaws):
    """直线视距不足时，从同一个避障最短路径场补充绕角出口，不穿越任何禁区。"""
    x, y, yaw = pose
    dc = cfg.decision
    result = []
    for k in range(dc.n_candidate_dirs):
        direction = yaw + 2 * math.pi * k / dc.n_candidate_dirs
        gx, gy = x + dc.candidate_max_dist * math.cos(direction), y + dc.candidate_max_dist * math.sin(direction)
        goal = nearest_reachable(field, gx, gy)
        if goal is None:
            continue
        path = extract_path(field, goal)
        ex, ey = path[-1]
        direct = math.hypot(ex - x, ey - y)
        length = path_length(path)
        if direct < dc.candidate_min_dist or length > 2 * dc.candidate_max_dist:
            continue
        if length < direct + 2 * grid.res:
            continue  # 直线路径已由原有候选覆盖，只补充需要拐弯的路线。
        absolute = math.atan2(ey - y, ex - x)
        if any(abs(wrap(absolute - excluded)) < 2 * math.pi / dc.n_candidate_dirs
               for excluded in exclude_yaws):
            continue
        bearing = wrap(absolute - yaw)
        straight = _ray_free(grid, x, y, absolute, dc.candidate_max_dist, grid.wall_lethal)
        cand = Candidate(cid=0, bearing_deg=math.degrees(bearing), abs_yaw=absolute,
                         endpoint=(ex, ey), straight_m=float(straight), path=path,
                         in_view=abs(bearing) <= math.radians(cfg.sensor.hfov_deg) / 2)
        _annotate(cand, grid, cfg)
        if len(visited_xy):
            cand.visited = bool(np.min(np.hypot(visited_xy[:, 0] - ex, visited_xy[:, 1] - ey)) < .75)
        if ground_fn is not None:
            for px, py in reversed(path):
                height = ground_fn(px, py)
                if not math.isnan(height):
                    cand.elevation_change_m = height - feet_h
                    break
        result.append(cand)
    return result


def _merge_same_way(cands: List[Candidate], tol_m: float = 0.6) -> List[Candidate]:
    """去掉"同一条路"的重复候选：短候选的终点离长候选的路线不到 tol_m，说明它只是同一条路上的一个点。
    直走廊里 0° 和 15° 常被拆成两个候选，不合并的话门控永远认为"有两个选择"，从不跳过调用。"""
    with_path = sorted([c for c in cands if c.path is not None], key=lambda c: -c.distance_m)
    kept = []
    for c in with_path:
        ex, ey = c.path[-1]
        same = any(np.min(np.hypot(np.asarray(k.path)[:, 0] - ex, np.asarray(k.path)[:, 1] - ey)) < tol_m
                   for k in kept)
        if not same:
            kept.append(c)
    return kept + [c for c in cands if c.path is None]


def _annotate(c: Candidate, grid: PlanGrid, cfg: AgentCfg):
    cc = cfg.cost
    p = np.asarray(c.path)
    c.distance_m = path_length(c.path)
    i, j = grid.to_cell(p[:, 0], p[:, 1])
    ok = grid.inside(i, j)
    hd = grid.human_dist[i[ok], j[ok]]
    if hd.size:
        c.min_clearance_to_human_m = float(min(hd.min(), 99.0))
    social = cc.agent_radius + cc.human_margin + cc.human_soft_radius
    c.crosses_human_zone = c.min_clearance_to_human_m < social
    unk = grid.unknown[i[ok], j[ok]]
    ei, ej = grid.to_cell(*c.path[-1])
    end_unknown = bool(grid.inside(ei, ej) and grid.unknown[ei, ej])
    c.leads_to_unknown = end_unknown or (unk.size > 0 and unk.mean() > 0.3)
