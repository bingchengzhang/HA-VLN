"""建图模块单元测试：用解析方法合成深度图，结果有标准答案。

运行：python3 tests/test_mapping.py   （也兼容 pytest）
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from havln_agent.config import AgentCfg
from havln_agent.mapping import (backproject, points_to_scan, depth_to_scan, StaticLayer, denormalize_depth,
                                 build_costmap, OCCUPIED, FREE)


def synth_depth(cfg, wall_dist=None, pitch=0.0, size=None):
    """合成一张深度图（米）：水平地面 + 可选的一面正对的墙（forward = wall_dist）。

    对每个像素的相机射线 (x, y, 1)·z 求它与地面、墙的交点，取最近的 z。
    """
    s = cfg.sensor
    w = h = size or s.img_width
    f = cfg.focal_px * (w / s.img_width)
    rows, cols = np.mgrid[0:h, 0:w]
    xr = (cols - (w - 1) / 2) / f
    yd = (rows - (h - 1) / 2) / f
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd_per_z = cp + yd * sp            # 每单位 z 的水平前向分量
    up_per_z = sp - yd * cp             # 每单位 z 的高度变化
    z = np.full((h, w), s.max_depth)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_floor = np.where(up_per_z < 0, -s.camera_height / up_per_z, np.inf)
        z = np.minimum(z, z_floor)
        if wall_dist is not None:
            z_wall = np.where(fwd_per_z > 0, wall_dist / fwd_per_z, np.inf)
            z = np.minimum(z, z_wall)
    z = np.clip(z, 0, s.max_depth)
    return z / s.max_depth if s.depth_normalized else z


def test_flat_wall_range():
    """正对 2 m 处的墙：每条射线的长度应是 2/cos(方位角)，且都判为命中。"""
    cfg = AgentCfg()
    scan = depth_to_scan(synth_depth(cfg, wall_dist=2.0), cfg)
    expect = 2.0 / np.cos(np.abs(scan.bearings))
    assert scan.is_hit[scan.valid].all()
    err = np.abs(scan.ranges - expect) / expect
    assert err[scan.valid].max() < 0.04, err.max()


def test_pitch_down_still_sees_wall():
    """低头 30°：画面里地面变多，但墙的水平距离不应变。"""
    cfg = AgentCfg()
    pitch = math.radians(-30)
    scan = depth_to_scan(synth_depth(cfg, wall_dist=2.0, pitch=pitch), cfg, pitch=pitch)
    hit = scan.valid & scan.is_hit
    assert hit.sum() > 0.8 * len(scan.ranges)
    expect = 2.0 / np.cos(np.abs(scan.bearings[hit]))
    assert (np.abs(scan.ranges[hit] - expect) / expect).max() < 0.05


def test_open_floor_no_false_hit():
    """空旷地面、深度截断处：不应产生任何障碍，射线只证明前方是空的。"""
    cfg = AgentCfg()
    scan = depth_to_scan(synth_depth(cfg, wall_dist=None), cfg)
    assert not scan.is_hit.any()
    assert scan.valid.all() and (scan.ranges > 3.0).all()


def test_human_pixels_block_but_do_not_hit():
    """被人体掩码覆盖的列：射线到人为止，但不算静态障碍。"""
    cfg = AgentCfg()
    depth = synth_depth(cfg, wall_dist=4.0)
    d_m = depth * cfg.sensor.max_depth
    mask = np.zeros_like(d_m, dtype=bool)
    mask[60:200, 118:138] = True                     # 画面中央一个人
    d_m[mask] = 1.5                                  # 人站在 1.5 m 处
    scan = points_to_scan(backproject(d_m, cfg, 0.0), cfg, human_mask=mask)
    center = np.abs(scan.bearings) < math.radians(2)
    assert center.any()
    assert not scan.is_hit[center].any()
    assert (np.abs(scan.ranges[center] - 1.5) < 0.1).all()
    side = np.abs(scan.bearings) > math.radians(20)
    assert scan.is_hit[side].all()


# ---------------------------------------------------------------- 高程图
from tests.fixtures.box_scene import BoxScene, stairs_scene, stairs_blocks
from havln_agent.mapping import UNKNOWN


def observe(layer, scene, cfg, poses):
    """依次站在 poses 上各看一眼，写进高程图。pose 可带第四个分量：俯仰角（低头为负）。"""
    for pose in poses:
        pitch = pose[3] if len(pose) > 3 else 0.0
        pose = pose[:3]
        depth, inst, _ = scene.render(pose, pitch)
        pts = backproject(denormalize_depth(depth, cfg), cfg, pitch)
        feet = scene.feet(pose)
        layer.update(pose, pts, feet, (inst > 0)[pts.rows, pts.cols])
        layer.mark_footprint(pose[0], pose[1], feet)


def _classified(layer, x, y, pad=2.0):
    """在 (x, y) 周围 pad 米的窗口上分类（台阶屏障要看邻格，不能只取一格）。返回 (栅格, 窗口起点)。"""
    i0, j0 = layer.world_to_cell(x - pad, y - pad)
    i1, j1 = layer.world_to_cell(x + pad, y + pad)
    i0, j0 = max(int(i0), 0), max(int(j0), 0)
    return layer.classify(i0, int(i1) + 1, j0, int(j1) + 1), i0, j0


def cell_class(layer, x, y):
    g, i0, j0 = _classified(layer, x, y)
    i, j = layer.world_to_cell(x, y)
    return int(g[i - i0, j - j0])


def window_class(layer, x, y, r=0.06):
    """(x, y) 附近 r 以内各格的分类。"""
    g, i0, j0 = _classified(layer, x, y)
    a0, b0 = layer.world_to_cell(x - r, y - r)
    a1, b1 = layer.world_to_cell(x + r, y + r)
    return g[a0 - i0:a1 - i0 + 1, b0 - j0:b1 - j0 + 1]


def corridor(cfg, blocks=(), length=8.0):
    walls = [((-1.0, -0.8), (length, -0.8)), ((-1.0, 0.8), (length, 0.8)),
             ((-1.0, -0.8), (-1.0, 0.8)), ((length, -0.8), (length, 0.8))]
    return BoxScene(cfg, walls, blocks=blocks)


def test_wall_occupied_floor_free():
    cfg = AgentCfg()
    scene = corridor(cfg, length=3.0)
    layer = StaticLayer(cfg)
    observe(layer, scene, cfg, [(0.0, 0.0, 0.0)])
    assert (window_class(layer, 3.0, 0.0) == OCCUPIED).any(), "正前方 3 m 的墙没被标成障碍"
    assert cell_class(layer, 2.0, 0.0) == FREE
    assert cell_class(layer, 1.5, 0.79) == OCCUPIED or (window_class(layer, 1.5, 0.8) == OCCUPIED).any()


def test_stairs_up_are_walkable():
    """上行楼梯：台阶面是可站立的地面，高度逐级升高，不能被当成墙。"""
    cfg = AgentCfg()
    scene = stairs_scene(cfg)
    layer = StaticLayer(cfg)
    observe(layer, scene, cfg, [(0.0, 0.0, 0.0), (0.75, 0.0, 0.0), (1.5, 0.0, 0.0)])
    for k in (0, 1, 2, 3):                              # 更高的台阶接近相机高度，从楼下看不全，属于正常
        x = 2.0 + k * 0.28 + 0.14                       # 第 k 级台阶面中央
        assert cell_class(layer, x, 0.0) == FREE, f"第 {k} 级台阶被判为 {cell_class(layer, x, 0.0)}"
        assert abs(layer.ground_height(x, 0.0) - (k + 1) * 0.18) < 0.03
    assert (window_class(layer, 3.0, 0.8) == OCCUPIED).any(), "楼梯两侧的墙应仍是障碍"


def test_table_and_box_are_obstacles():
    """悬空桌面（下面能看到地面）与实心矮柜（0.5 m）都必须是障碍；0.15 m 的门槛可以跨过去。"""
    cfg = AgentCfg()
    blocks = [(2.0, -0.3, 2.6, 0.3, 0.70, 0.75),        # 桌面
              (4.0, -0.3, 4.4, 0.3, 0.0, 0.50),         # 矮柜
              (5.5, -0.8, 5.6, 0.8, 0.0, 0.15)]         # 门槛
    scene = corridor(cfg, blocks)
    layer = StaticLayer(cfg)
    observe(layer, scene, cfg, [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, -0.5, 0.0), (3.2, 0.5, 0.0)])
    assert (window_class(layer, 2.3, 0.0, 0.25) == OCCUPIED).any(), "桌子没被标成障碍"
    assert (window_class(layer, 4.2, 0.0, 0.15) == OCCUPIED).any(), "矮柜没被标成障碍"
    assert not (window_class(layer, 5.55, 0.0, 0.04) == OCCUPIED).any(), "0.15 m 的门槛不该挡路"


def test_ledge_blocked_when_close():
    """站在 1 m 高的平台上，平台在 x=2 处断开：走近并低头看一眼后，边缘必须是障碍。"""
    cfg = AgentCfg()
    scene = corridor(cfg, blocks=[(-1.0, -0.8, 2.0, 0.8, 0.0, 1.0)])
    layer = StaticLayer(cfg)
    down = -math.radians(cfg.sensor.tilt_deg)
    observe(layer, scene, cfg, [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (1.5, 0.0, 0.0, down)])
    assert (window_class(layer, 1.95, 0.0, 0.1) == OCCUPIED).any(), "平台边缘没被标成障碍"


def test_stairs_down_walkable_from_top():
    """站在二层平台上往下看楼梯：走近楼梯口并低头看一眼后，下行台阶不应被当成断崖。"""
    cfg = AgentCfg()
    # 楼梯沿 −x 方向下降：平台在 x∈[4,7]，高 1.8；台阶从 x=4 往 x=1.2 逐级降低
    blocks = [(4.0, -0.8, 7.0, 0.8, 0.0, 1.8)] + \
             [(4.0 - (k + 1) * 0.28, -0.8, 4.0, 0.8, 0.0, 1.8 - (k + 1) * 0.18) for k in range(9)]
    scene = corridor(cfg, blocks)
    layer = StaticLayer(cfg)
    down = -math.radians(cfg.sensor.tilt_deg)
    observe(layer, scene, cfg, [(6.0, 0.0, math.pi), (5.0, 0.0, math.pi), (4.5, 0.0, math.pi),
                                (4.5, 0.0, math.pi, down)])
    for k in (1, 3):
        x = 4.0 - (k + 1) * 0.28 + 0.14
        assert cell_class(layer, x, 0.0) == FREE, f"下行第 {k + 1} 级被判为 {cell_class(layer, x, 0.0)}"


def test_costmap_inflation():
    """墙前硬禁区 = agent 半径 + 墙余量；人的禁区 = 人体点外 agent 半径 + 人余量。"""
    cfg = AgentCfg()
    scene = corridor(cfg, length=3.0)
    layer = StaticLayer(cfg)
    observe(layer, scene, cfg, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    cm = build_costmap(layer, (0.0, 0.0), np.array([[1.0, 0.3]]), [], cfg)
    r_wall = cfg.cost.agent_radius + cfg.cost.wall_margin
    _, j0 = cm.world_to_cell(0.0, 0.0)
    i, _ = cm.world_to_cell(3.0 - r_wall - 0.12, 0.0)
    assert not cm.lethal[i, j0]
    i, _ = cm.world_to_cell(3.0 - r_wall + 0.03, 0.0)
    assert cm.lethal[i, j0] and cm.wall_lethal[i, j0]
    r_h = cfg.cost.agent_radius + cfg.cost.human_margin
    ii, jj = cm.world_to_cell(1.0, 0.3 - r_h + 0.06)
    assert cm.human_lethal[ii, jj]
    ii, jj = cm.world_to_cell(1.0, 0.3 - r_h - 0.08)
    assert not cm.human_lethal[ii, jj]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if failed else 0)
