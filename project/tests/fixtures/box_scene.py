"""单元测试用的合成场景：竖直墙面、实心方块（平台、台阶、桌子）和行人圆柱，按针孔相机光线求交渲染深度图。

只用于在没有 Habitat 的机器上检查代码逻辑（坐标、反投影、高程图、规划、决策流程是否自洽），
不代表 Habitat 的真实效果。相机参数、动作步长与 AgentCfg 一致。
可通行规则模仿导航网格：台阶高差 ≤ 0.2 m 能走上去/走下来，更高就被挡住（计一次碰撞）。
"""
import math

import numpy as np

WALL_TOP = 6.0
MAX_CLIMB = 0.2


class BoxScene:
    def __init__(self, cfg, walls, humans=(), blocks=(), agent_radius=0.1, human_radius=0.25):
        """walls:  [((x0, y0), (x1, y1)), ...] 竖直墙段，从地面一直到很高
        blocks: [(x0, y0, x1, y1, z_bottom, z_top), ...] 轴对齐方块；z_bottom=0 是实心台子，
                z_bottom>0 是悬空的板（如桌面，下面能看到地面）
        humans: [(x, y), ...] 站在地面 z=0 上的静止行人"""
        self.cfg = cfg
        # 竖直面：(端点a, 端点b, 底高, 顶高, 是否属于实心块)。实心块的侧面不参与水平碰撞，
        # 能不能走上去只看脚下高度差（模仿导航网格把楼梯平滑成斜坡）
        self.walls = [(np.array(a, float), np.array(b, float), 0.0, WALL_TOP, False) for a, b in walls]
        self.blocks = [tuple(map(float, b)) for b in blocks]
        for x0, y0, x1, y1, zb, zt in self.blocks:
            for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                self.walls.append((np.array(a), np.array(b), zb, zt, zb <= 1e-6))
        self.humans = [np.array(h, float) for h in humans]
        self.agent_r, self.human_r = agent_radius, human_radius
        self.collisions = 0

    def floor_h(self, x, y):
        """脚下地面高度：站在实心方块上就是块顶，否则是 0。悬空的板不能站（从下面钻过去）。"""
        h = 0.0
        for x0, y0, x1, y1, zb, zt in self.blocks:
            if zb <= 1e-6 and x0 <= x <= x1 and y0 <= y <= y1:
                h = max(h, zt)
        return h

    # ---------- 渲染 ----------
    def render(self, pose, pitch=0.0):
        """返回 (归一化深度 (H, W, 1), 人体实例图 (H, W), RGB (h, w, 3))。相机高度 = 脚下高度 + camera_height。"""
        s = self.cfg.sensor
        w = h = s.img_width
        f = self.cfg.focal_px
        rows, cols = np.mgrid[0:h, 0:w]
        xr = (cols - (w - 1) / 2) / f
        yd = (rows - (h - 1) / 2) / f
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = cp + yd * sp
        up = sp - yd * cp
        left = -xr
        x, y, yaw = pose
        dx = fwd * math.cos(yaw) - left * math.sin(yaw)
        dy = fwd * math.sin(yaw) + left * math.cos(yaw)
        cz = self.floor_h(x, y) + s.camera_height

        z = np.full((h, w), s.max_depth)
        inst = np.zeros((h, w), dtype=np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.minimum(z, np.where(up < 0, -cz / up, np.inf))                 # 地面 z=0
            for x0, y0, x1, y1, zb, zt in self.blocks:                              # 块的顶面和底面
                for zz in (zt, zb) if zb > 1e-6 else (zt,):
                    t = (zz - cz) / up
                    px, py = x + t * dx, y + t * dy
                    ok = (t > 0) & (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
                    z = np.where(ok & (t < z), t, z)
            for a, b, z0, z1, _ in self.walls:
                e = b - a
                den = dx * e[1] - dy * e[0]
                rx, ry = a[0] - x, a[1] - y
                t = (rx * e[1] - ry * e[0]) / den
                u = (rx * dy - ry * dx) / den
                hz = cz + t * up
                ok = (t > 0) & (u >= 0) & (u <= 1) & (hz >= z0) & (hz <= z1)
                z = np.where(ok & (t < z), t, z)
            for k, c in enumerate(self.humans, start=1):
                ox, oy = x - c[0], y - c[1]
                A = dx * dx + dy * dy
                B = 2 * (ox * dx + oy * dy)
                C = ox * ox + oy * oy - self.human_r ** 2
                disc = B * B - 4 * A * C
                t = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A)
                hz = cz + t * up
                ok = (disc >= 0) & (t > 0) & (hz >= 0) & (hz <= 1.7)
                hit = ok & (t < z)
                z = np.where(hit, t, z)
                inst = np.where(hit, k, inst)
        z = np.clip(z, 0, s.max_depth)
        depth = (z / s.max_depth)[..., None].astype(np.float32)
        shade = (255 * (1 - np.clip(z / s.max_depth, 0, 1))).astype(np.uint8)
        rgb = np.stack([shade] * 3, axis=-1)[:s.rgb_height, :s.rgb_width]
        return depth, inst, rgb

    # ---------- 运动 ----------
    def clearance(self, x, y, feet):
        """到最近的"挡身体"的竖直面的水平距离：只算覆盖脚下 +0.2 m 到 +1.5 m 之间的面。"""
        d = math.inf
        p = np.array([x, y])
        for a, b, z0, z1, solid in self.walls:
            if solid or z1 <= feet + MAX_CLIMB or z0 >= feet + 1.5:
                continue
            e = b - a
            u = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
            d = min(d, float(np.linalg.norm(p - (a + u * e))))
        return d

    def human_clear(self, x, y):
        return min((float(np.hypot(x - c[0], y - c[1])) - self.human_r for c in self.humans), default=math.inf)

    def step(self, pose, action):
        x, y, yaw = pose
        s = self.cfg.sensor
        if action == "TURN_LEFT":
            return (x, y, yaw + math.radians(s.turn_deg))
        if action == "TURN_RIGHT":
            return (x, y, yaw - math.radians(s.turn_deg))
        if action != "MOVE_FORWARD":
            return pose
        nx, ny = x + s.step_len * math.cos(yaw), y + s.step_len * math.sin(yaw)
        f0, f1 = self.floor_h(x, y), self.floor_h(nx, ny)
        blocked = (abs(f1 - f0) > MAX_CLIMB + 1e-6 or self.clearance(nx, ny, f1) < self.agent_r
                   or self.human_clear(nx, ny) < self.agent_r)
        if blocked:
            self.collisions += 1
            return pose
        return (nx, ny, yaw)

    def feet(self, pose):
        return self.floor_h(pose[0], pose[1])


class FixtureSegmenter:
    """测试用：直接返回渲染时得到的人体实例图（相当于完美检测）。"""

    def __init__(self):
        self.next_inst = None

    def segment(self, rgb, out_shape):
        return self.next_inst, set()


def corridor_scene(cfg, humans=()):
    """L 形走廊：从 (0,0) 出发向 +x 走 6 m，在尽头左转（+y）再走 4 m。走廊宽 1.6 m。"""
    walls = [
        ((-1.0, -0.8), (6.8, -0.8)),
        ((-1.0, 0.8), (5.2, 0.8)),
        ((6.8, -0.8), (6.8, 6.0)),
        ((5.2, 0.8), (5.2, 6.0)),
        ((-1.0, -0.8), (-1.0, 0.8)),
        ((5.2, 6.0), (6.8, 6.0)),
    ]
    return BoxScene(cfg, walls, humans)


def stairs_blocks(x_start, n_steps, rise=0.18, run=0.28, y0=-0.8, y1=0.8, top_len=3.0):
    """沿 +x 上升的楼梯：第 k 级是从 x_start + k·run 开始一直延伸到楼梯顶平台末端的实心块，
    最后一级之后接一个长 top_len 的平台（与最后一级同高）。"""
    end = x_start + n_steps * run + top_len
    return [(x_start + k * run, y0, end, y1, 0.0, (k + 1) * rise) for k in range(n_steps)]


def stairs_scene(cfg, n_steps=10):
    """直走廊尽头接一段上行楼梯：x∈[2, 4.8] 是 10 级台阶，之后是高 1.8 m 的二层平台，长 3 m，尽头是墙。"""
    end = 2.0 + n_steps * 0.28 + 3.0
    walls = [((-1.0, -0.8), (end, -0.8)), ((-1.0, 0.8), (end, 0.8)),
             ((-1.0, -0.8), (-1.0, 0.8)), ((end, -0.8), (end, 0.8))]
    return BoxScene(cfg, walls, blocks=stairs_blocks(2.0, n_steps))
