"""坐标约定与换算。

地图平面坐标 (x, y)：x = habitat 世界 x，y = −habitat 世界 z（俯视时逆时针为正）。
yaw：以 +x 为 0、逆时针为正。TURN_LEFT 使 yaw 增加 15°。
agent 坐标系：forward 向前、left 向左。
"""
import math

import numpy as np


def wrap(a):
    """角度归一到 [-pi, pi)。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def habitat_to_map(position, rotation):
    """habitat 的 position (x, y, z) 与 rotation 四元数 → 地图位姿 (x, y, yaw) 和地面高度。

    rotation 可以是 numpy-quaternion 对象，也可以是 [x, y, z, w] 列表。
    agent 的正前方在本体系里是 −z。
    """
    q = _as_wxyz(rotation)
    fwd = _quat_rotate(q, np.array([0.0, 0.0, -1.0]))
    x, y_up, z = float(position[0]), float(position[1]), float(position[2])
    yaw = math.atan2(-fwd[2], fwd[0])
    return (x, -z, yaw), y_up


def agent_to_world(pose, forward, left):
    """agent 坐标系下的点 → 地图坐标（支持 numpy 数组）。"""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + forward * c - left * s, y + forward * s + left * c


def world_to_agent(pose, wx, wy):
    x, y, yaw = pose
    dx, dy = np.asarray(wx) - x, np.asarray(wy) - y
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c + dy * s, -dx * s + dy * c


def bearing_to(pose, wx, wy):
    """从 agent 看点 (wx, wy) 的方位角，正前方为 0、向左为正。"""
    f, l = world_to_agent(pose, wx, wy)
    return math.atan2(float(l), float(f))


# ---------- 四元数（不依赖 numpy-quaternion，单元测试也能跑）----------
def _as_wxyz(q):
    if hasattr(q, "w") and hasattr(q, "x"):
        return np.array([q.w, q.x, q.y, q.z], dtype=float)
    q = np.asarray(q, dtype=float)
    if q.shape == (4,):
        return np.array([q[3], q[0], q[1], q[2]])   # habitat 的 coeffs 顺序是 [x, y, z, w]
    raise ValueError(f"无法识别的四元数: {q}")


def _quat_rotate(q_wxyz, v):
    w, x, y, z = q_wxyz
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def yaw_to_habitat_quat_xyzw(yaw):
    """地图 yaw → habitat 绕 +y 轴的旋转四元数 [x, y, z, w]（生成训练数据时设定朝向用）。

    yaw = 0 时 agent 面向 +x，即绕 y 轴转 −90°（habitat 默认面向 −z）。
    """
    angle = yaw - math.pi / 2
    return [0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]
