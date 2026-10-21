"""HA-VLN 2.0 环境适配器（只能在官方镜像里运行）。

环境的建法照搬官方 export_cma_submission.py：同一份 /app/config/challenge_submission.yaml、
同一个 HAVLNCEDaggerEnv、同一套数据路径。区别只有两点：
  - 传感器保留 RGB 和深度（官方脚本只为 CMA 关掉了度量）
  - 本地评测时可以打开 DISTANCE_TO_GOAL / SUCCESS / COLLISIONS_DETAIL 三个度量
这些度量只用来打分，智能体看不到。

PrivilegedView 汇集了所有特权信息（参考路径、终点、行人真实位置），
只给评测脚本和训练数据生成脚本用，智能体代码里不出现。
"""
import math
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..geometry import habitat_to_map

ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN")
# 官方镜像里在 /app/config；不用 Docker 的原生环境，把镜像里的 /app/config 拷出来，用环境变量指过去
CHALLENGE_CONFIG = os.environ.get("HAVLN_CHALLENGE_CONFIG", "/app/config/challenge_submission.yaml")
EVAL_MEASURES = ["DISTANCE_TO_GOAL", "SUCCESS", "COLLISIONS_DETAIL"]


def import_runtime(source_root: Path):
    """与官方 policy_runtime() 相同的导入方式：先用镜像里打过补丁的 Habitat，再把公开源码挂到 vlnce_baselines 下。"""
    import habitat
    import habitat_sim
    if habitat.__version__ != "0.1.7" or habitat_sim.__version__ != "0.1.7":
        raise RuntimeError("必须在官方 HA-VLN 2.0 镜像（Habitat 0.1.7）里运行")
    import habitat_extensions  # noqa: F401  注册传感器和度量
    import vlnce_baselines
    import vlnce_baselines.common
    baseline = source_root / "agent/VLN-CE/vlnce_baselines"
    if str(baseline) not in vlnce_baselines.__path__:
        vlnce_baselines.__path__.append(str(baseline))
        vlnce_baselines.common.__path__.append(str(baseline / "common"))
    try:                                   # COLLISIONS_DETAIL 等度量定义在 HASimulator 里
        import sys
        sys.path.insert(0, str(source_root))
        import HASimulator.measures  # noqa: F401
    except Exception:
        pass


def make_env(data_root: str, split: str, episode_ids: List[str], gpu: int, source_root: str,
             with_metrics: bool = False, extra_sensors=()):
    """extra_sensors：额外打开的传感器（如 SEMANTIC_SENSOR），只给离线诊断用，正式 rollout 不传。
    额外传感器的分辨率、视场角、安装位置一律复制 RGB_SENSOR，保证像素对齐。"""
    from habitat import Config  # noqa: F401
    import_runtime(Path(source_root))
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.environments import HAVLNCEDaggerEnv

    if not Path(CHALLENGE_CONFIG).is_file():
        raise FileNotFoundError(
            f"找不到比赛配置 {CHALLENGE_CONFIG}。原生环境请把官方镜像的 /app/config 拷出来，"
            f"并 export HAVLN_CHALLENGE_CONFIG=<拷出目录>/challenge_submission.yaml（见 docs/TRAINING.md 第 3 步）")
    config = get_config(CHALLENGE_CONFIG)
    config.defrost()
    config.EVAL.SPLIT = split
    cfg = config.TASK_CONFIG
    cfg.DATASET.SPLIT = split
    cfg.DATASET.DATA_PATH = data_root + "/HA-R2R/{split}/{split}.json.gz"
    cfg.DATASET.SCENES_DIR = data_root + "/scene_datasets"
    cfg.DATASET.EPISODES_ALLOWED = list(episode_ids)
    cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu
    cfg.SIMULATOR.HUMAN_GLB_PATH = data_root + "/HAPS2_0"
    cfg.SIMULATOR.HUMAN_INFO_PATH = data_root + "/Multi-Human-Annotations/human_motion.json"
    cfg.SIMULATOR.RECOMPUTE_NAVMESH_PATH = data_root + "/recompute_navmesh"
    sensors = list(cfg.SIMULATOR.AGENT_0.SENSORS)
    for s in ("RGB_SENSOR", "DEPTH_SENSOR"):
        if s not in sensors:
            sensors.append(s)
    for s in extra_sensors:
        sc, rgb = getattr(cfg.SIMULATOR, s), cfg.SIMULATOR.RGB_SENSOR
        for k in ("WIDTH", "HEIGHT", "HFOV", "POSITION", "ORIENTATION"):
            if hasattr(rgb, k):
                setattr(sc, k, getattr(rgb, k))
        if s not in sensors:
            sensors.append(s)
    cfg.SIMULATOR.AGENT_0.SENSORS = sensors
    cfg.TASK.SENSORS = ["INSTRUCTION_SENSOR"]
    cfg.TASK.MEASUREMENTS = EVAL_MEASURES if with_metrics else []
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    cfg.SEED = 0
    config.freeze()
    if list(cfg.TASK.POSSIBLE_ACTIONS) != list(ACTIONS):
        raise RuntimeError(f"镜像动作集与预期不同：{cfg.TASK.POSSIBLE_ACTIONS}")
    env = HAVLNCEDaggerEnv(config=config)
    return env


def sensor_report(env) -> dict:
    """打印实际的传感器参数，用来核对 config.py 里标了 VERIFY 的值。"""
    sim_cfg = env.habitat_env._config.SIMULATOR
    out = {}
    for name in ("RGB_SENSOR", "DEPTH_SENSOR"):
        s = getattr(sim_cfg, name)
        out[name] = {k: getattr(s, k) for k in ("WIDTH", "HEIGHT", "HFOV", "POSITION") if hasattr(s, k)}
    d = sim_cfg.DEPTH_SENSOR
    out["DEPTH_SENSOR"].update({k: getattr(d, k) for k in ("MIN_DEPTH", "MAX_DEPTH", "NORMALIZE_DEPTH") if hasattr(d, k)})
    out["TILT_ANGLE"] = getattr(sim_cfg, "TILT_ANGLE", None)
    out["ALLOW_SLIDING"] = sim_cfg.HABITAT_SIM_V0.ALLOW_SLIDING
    out["ACTIONS"] = list(env.habitat_env._config.TASK.POSSIBLE_ACTIONS)
    return out


class EpisodeRunner:
    """把环境观测整理成智能体需要的东西：rgb、depth、pose、脚下高度、t。"""

    def __init__(self, env):
        self.env = env
        self.t0 = time.monotonic()

    @property
    def sim(self):
        return self.env.habitat_env.sim

    def pose(self):
        st = self.sim.get_agent_state()
        pose, _ = habitat_to_map(st.position, st.rotation)
        return pose

    def height(self):
        """agent 脚下的世界高度（habitat 的 position[1]），上下楼时会变。"""
        return float(self.sim.get_agent_state().position[1])

    def reset(self):
        obs = self.env.reset()
        self.t0 = time.monotonic()      # 行人动画从 reset 起按墙钟走，t 与之对齐
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        return obs, done, info

    def now(self):
        return time.monotonic() - self.t0

    @staticmethod
    def instruction(obs, episode):
        ins = obs.get("instruction", {})
        if isinstance(ins, dict) and ins.get("text"):
            return ins["text"]
        return episode.instruction.instruction_text


class PrivilegedView:
    """特权信息。只给评测和训练数据生成用。"""

    def __init__(self, env):
        self.env = env

    @property
    def episode(self):
        return self.env.current_episode

    def goal_xy(self):
        g = self.episode.goals[0].position
        return (float(g[0]), -float(g[2]))

    def reference_path_xy(self):
        return [(float(p[0]), -float(p[2])) for p in self.episode.reference_path]

    def reference_path_xyh(self):
        """参考路径：(地图 x, 地图 y, 高度)。"""
        return [(float(p[0]), -float(p[2]), float(p[1])) for p in self.episode.reference_path]

    def goal_xyh(self):
        g = self.episode.goals[0].position
        return (float(g[0]), -float(g[2]), float(g[1]))

    def human_positions_xy(self):
        hp = getattr(self.env.habitat_env.sim, "_human_posisions", {}) or {}
        return [(float(p[0]), -float(p[2])) for p, _ in hp.values()]

    def distance_to_goal(self):
        """与官方 DistanceToGoal 一致：三维欧氏距离（楼层不对时高度差也算进去）。"""
        st = self.env.habitat_env.sim.get_agent_state()
        g = self.episode.goals[0].position
        return math.dist([float(v) for v in st.position], [float(v) for v in g])
