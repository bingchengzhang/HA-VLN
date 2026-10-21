"""全部可调参数集中在这里。

分两类，对应架构里的原则：
  - 物理类：由仿真器/任务规则决定，查实后就不再动
  - 猜测类：对任务统计规律的假设，交给 scripts/param_search.py 自动搜索

注释里的"源码"指 HA-VLN 公开仓库（UWMILab/HA-VLN）里已经查到的值；
标了 VERIFY 的需要进官方镜像核对（/app/config/challenge_submission.yaml、/opt/havln-challenge/scoring）。
"""
from dataclasses import dataclass, field, asdict, fields, is_dataclass
import json
import math


@dataclass
class SensorCfg:
    """相机与动作（物理类）。"""
    img_width: int = 256              # 源码 vlnce_task.yaml：DEPTH 256x256
    img_height: int = 256
    rgb_width: int = 224              # 源码：RGB 224x224
    rgb_height: int = 224
    hfov_deg: float = 90.0            # 源码：HFOV 90
    camera_height: float = 1.25       # habitat 0.1.7 传感器默认 POSITION [0, 1.25, 0]；VERIFY
    max_depth: float = 10.0           # habitat 默认 MAX_DEPTH；VERIFY
    depth_normalized: bool = True     # habitat 默认 NORMALIZE_DEPTH；VERIFY
    step_len: float = 0.25            # 源码：FORWARD_STEP_SIZE
    turn_deg: float = 15.0            # 源码：TURN_ANGLE
    tilt_deg: float = 30.0            # 比赛配置实测 TILT_ANGLE=30（sensor_report.json，2026-10-18）


@dataclass
class MapCfg:
    """静态层（每层楼一张 2.5D 高程图）。"""
    resolution: float = 0.05
    size_m: float = 48.0              # 以起点为中心；R2R 路径一般 10 m 量级，留足余量
    obstacle_min_h: float = 0.10      # 虚拟激光与行人观测用：离脚下多高算"身体高度带"
    obstacle_max_h: float = 1.60      # 身体高度：高于地面这么多的东西撞不到（吊灯、门楣）
    n_virtual_rays: int = 128
    # 可通行性：与仿真器导航网格的参数对齐（VERIFY：镜像里 NavMeshSettings 的 agent_max_climb / agent_max_slope）
    max_step_h: float = 0.20          # 能跨上去的最大台阶高度
    height_tolerance_m: float = 0.005  # 深度反投影/栅格化的毫米级误差，不把临界台阶膨胀成墙
    # 跨过未观测缝隙比较高差时允许的坡度。比导航网格的 45° 严：相机水平时看不到 45° 以下，
    # 用 45° 推断就永远分不清断崖和陡坡；楼梯一般 30–37°，取 38° 仍能放行楼梯
    gap_slope_deg: float = 38.0
    max_gap_m: float = 1.5            # 跨缝比较的最大缝宽
    look_down_ahead_m: float = 1.0    # 前方这段路线穿过未观测区域时，低头看一眼（下楼、断崖、拐角）
    look_down_every_m: float = 1.5    # 两次低头之间至少走这么远
    floor_window_below: float = 1.5   # 只把脚下 −1.5 m 到 +1.6 m 之间的点写进本层的图
    floor_window_above: float = 1.6
    floor_switch_h: float = 1.5       # 脚下高度偏离本层参考高度超过它（连续几步）就换一层
    floor_switch_steps: int = 3


@dataclass
class CostCfg:
    """代价图膨胀。

    碰撞判定（源码 HASimulator）：行人以静态物体加入场景并重算 navmesh，agent 半径 0.1；
    碰撞计数来自 sim.previous_step_collided，即"这一步被 navmesh 挡住/滑动了"。
    按公开代码，擦墙也会被计为碰撞，所以墙的余量和人一样重要。VERIFY：官方计分镜像是否只计人。
    """
    agent_radius: float = 0.10                 # 源码：recompute navmesh agent_radius 0.1
    wall_margin: float = 0.12                  # 离墙至少 0.22 m，保证不触发滑动
    wall_soft_radius: float = 0.40
    human_margin: float = 0.25                 # 人体点云外再留多少（人体网格比深度看到的更宽）
    # 等太久后放宽到这个值，从人旁边挤过去（12 秒后人不再动，干等没有意义）。
    # 下限：仿真器在人体网格外按 agent 半径 0.1 m 判碰撞，我们的禁区是"人体可见表面外 agent_radius + 这个余量"，
    # 所以余量 ≥ 0 就不会物理碰撞；留 0.08 是给深度看不到的人体背面和姿态变化。不要调到 0.05 以下
    human_margin_relaxed: float = 0.08
    relax_steps: int = 24                      # 放宽持续多少步
    human_soft_radius: float = 0.80            # 社交软代价的延伸距离
    unseen_body_radius: float = 0.35           # 看不见的人用圆盘表示时的身体半径
    unknown_cost: float = 2.0
    soft_cost_max: float = 20.0
    window_m: float = 16.0                     # 只在 agent 周围这么大的窗口里算代价图


@dataclass
class HumanCfg:
    """行人层。"""
    assoc_dist: float = 0.8
    body_half_depth: float = 0.15     # 表面点到身体中心的补偿（沿视线方向）
    vel_smooth: float = 0.4
    static_speed: float = 0.05
    uncert_growth: float = 0.3        # 看不见的人，不确定半径每秒增长
    uncert_max: float = 1.5           # HAPS 中约九成行人位移 < 1.5 m
    track_ttl_s: float = 60.0         # 人停下后不会消失，记忆保留整个 episode 量级
    predict_horizon_s: float = 1.0
    motion_duration_s: float = 12.0   # 源码：120 帧 @ 10 Hz，之后静止
    min_mask_pixels: int = 80         # 太小的检测不要（远处或误检）
    det_conf: float = 0.35


@dataclass
class PlanCfg:
    """规划与执行。"""
    plan_resolution: float = 0.05     # 保留代价图窄缝；2x2 任意障碍池化会额外封闭可通行空间
    lookahead_m: float = 0.5
    turn_tol_deg: float = 7.5
    replan_every: int = 4             # 执行时每隔几步重算一次路径（路径被挡时立即重算）
    max_wait_steps: int = 10          # 紧急覆盖时最多原地等几步
    stuck_forward_m: float = 0.05     # 前进后位移小于它判为被挡
    slide_ratio: float = 0.8          # 平地上前进位移不到 step_len 的这个比例：沿墙滑动了（官方同样计一次碰撞）
    initial_spin: bool = True         # 开局转一圈初始化地图和视野记忆（24 步）


@dataclass
class DecisionCfg:
    """决策层（猜测类，交给自动搜索）。"""
    segment_max_m: float = 3.0
    junction_radius: float = 1.5
    junction_min_gap_deg: float = 30.0
    llm_fallback_steps: int = 40
    min_call_gap_steps: int = 4        # 两次模型调用至少隔几步（防止路线反复失效时连续调用）
    candidate_min_dist: float = 1.0
    candidate_max_dist: float = 3.0
    n_candidate_dirs: int = 24
    candidate_merge_deg: float = 45.0  # 连续可行方向超过这个宽度就拆成多个候选
    overrun_factor: float = 1.6        # 走过的距离超过"预期路径长度"的倍数就强制决策
    stop_confirm_max_m: float = 2.5    # 目标地标的深度距离小于它才真正 STOP
    stop_approach_steps: int = 16      # 地标太远时最多再靠近几步
    max_stop_rejects: int = 2          # 模型说"到了"但距离确认不通过的次数上限；超过后按模型的判断停下（记为未确认）
    max_observes: int = 6              # 每个 episode 最多"转身看"几次（每次 6–12 步）


@dataclass
class SlowCfg:
    """慢回路（大模型）。"""
    backend: str = "openai_compatible"     # DeepSeek API 与本地 vLLM 都走 OpenAI 兼容接口
    base_url: str = "http://zbc-vlm:8000/v1"      # scripts/serve_vlm.sh 起的容器，同一个 docker 网络里按名字访问
    model: str = "qwen-vl"                        # vLLM 的 --served-model-name
    api_key_env: str = "LLM_API_KEY"       # 从环境变量读，不写进代码
    supports_images: bool = True           # DeepSeek 设为 False，走纯文本
    temperature: float = 0.0
    max_tokens: int = 256                  # 决策点的 JSON 很短
    compile_max_tokens: int = 1024         # 指令拆解的子目标清单长：256 会被截断（服务器 20 条里 11 条拆解失败）
    timeout_s: float = 60.0
    retries: int = 1
    cache_dir: str = "/workspace/llm_cache"
    image_side: int = 336                  # 送给 VLM 的图片边长（Qwen2.5-VL 的 patch 是 28，336 = 12×28）
    use_env_proxy: bool = False            # 本地服务不走代理；调外部 API（DeepSeek）时设 True


@dataclass
class AgentCfg:
    sensor: SensorCfg = field(default_factory=SensorCfg)
    map: MapCfg = field(default_factory=MapCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    human: HumanCfg = field(default_factory=HumanCfg)
    plan: PlanCfg = field(default_factory=PlanCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)
    slow: SlowCfg = field(default_factory=SlowCfg)
    max_steps: int = 500              # 官方动作预算
    success_dist: float = 3.0         # 源码 SUCCESS_DISTANCE
    # 位姿直接读仿真器（待主办方确认是否允许）。不允许时再补航位推算，现在不做

    @property
    def focal_px(self) -> float:
        s = self.sensor
        return (s.img_width / 2.0) / math.tan(math.radians(s.hfov_deg) / 2.0)

    # ---------- 读写：一份 JSON 只写需要覆盖的字段 ----------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, overrides: dict):
        cfg = cls()
        _apply(cfg, overrides or {})
        return cfg

    @classmethod
    def from_json(cls, path):
        if not path:
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _apply(obj, overrides: dict, prefix=""):
    names = {f.name for f in fields(obj)}
    for key, value in overrides.items():
        if key not in names:
            raise KeyError(f"未知参数 {prefix}{key}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            _apply(cur, value, prefix + key + ".")
        else:
            setattr(obj, key, type(cur)(value) if cur is not None else value)


def set_by_path(cfg: AgentCfg, dotted: str, value):
    """param_search 用：set_by_path(cfg, 'decision.segment_max_m', 2.5)。"""
    *parents, leaf = dotted.split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, type(getattr(obj, leaf))(value))
