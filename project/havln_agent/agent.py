"""分层智能体：每步一次 act(rgb, depth, pose, t) → 六个动作之一。

快回路（每步）：深度 + 位姿 → 静态层；RGB → 人体分割 → 行人层；合成代价图；执行器出动作。
决策层（只在决策点）：生成候选 + 风险摘要 → 门控 → 大模型选择 → 新路线 / 子目标推进 / 停止。

只用到 RGB、深度、位姿和指令文本，不碰任何特权信息（参考路径、行人真实位置、导航网格）。
"""
import math
import time
from typing import Callable, Optional

import numpy as np

from .config import AgentCfg
from .executor import Executor, goal_depth, side_bearing, box_depth_bearing, FORWARD, LEFT, RIGHT, STOP, DOWN
from .geometry import wrap, world_to_agent
from .mapping import (StaticLayer, HumanLayer, backproject, points_to_scan, denormalize_depth,
                      observe_humans, build_costmap)
from .perception.human_seg import HumanSegmenter
from .planning.candidates import generate_candidates
from .planning.decision import DecisionTrigger, subgoal_may_complete, need_llm_call
from .planning.grid_search import make_plan_grid, shortest_paths, nearest_reachable, extract_path, path_length
from .slow.slow_loop import SlowLoop, PlanStore, Decision
from .slow.view_memory import ViewMemory, build_images


def truncate_path(path, max_len):
    """路线只取前 max_len 米：每段走完就是一个决策点。"""
    if len(path) < 2:
        return path
    out, acc = [path[0]], 0.0
    for a, b in zip(path[:-1], path[1:]):
        acc += math.hypot(b[0] - a[0], b[1] - a[1])
        out.append(b)
        if acc >= max_len:
            break
    return out


class HierarchicalAgent:
    def __init__(self, cfg: AgentCfg, segmenter: HumanSegmenter, slow: SlowLoop, plans: PlanStore,
                 path_stats: Optional[dict] = None,
                 decision_override: Optional[Callable] = None):
        """decision_override(agent, cands, ctx) -> Decision：由专家代替大模型做选择（造训练数据、逐层诊断）。
        diag_oracle：只记录诊断（每个决策点的候选覆盖与专家答案），不影响动作；本地评测时由 rollout 设置。"""
        self.diag_oracle = None
        self.cfg = cfg
        self.seg = segmenter
        self.slow = slow
        self.plans = plans
        self.path_stats = path_stats or {}
        self.decision_override = decision_override

    # ------------------------------------------------------------------ episode
    def reset(self, episode_id, instruction: str, pose, height: float = 0.0):
        """height：agent 脚下的世界高度（habitat 的 position[1]），用于楼梯和分楼层建图。"""
        c = self.cfg
        self.episode_id, self.instruction = str(episode_id), instruction
        self.plan = self.plans.get(episode_id, instruction, self.slow.client)
        self.sub_idx = 0
        self.floors = [{"ref": height, "layer": StaticLayer(c, origin_xy=pose[:2])}]
        self.floor = 0
        self.feet = height
        self._off_floor = 0
        self.humans = HumanLayer(c)
        self.exe = Executor(c)
        self.memory = ViewMemory()
        n_words = len(instruction.split())
        expected = None
        if "a" in self.path_stats:
            expected = self.path_stats["a"] + self.path_stats["b"] * n_words
        self.trig = DecisionTrigger(c, expected)
        self.step = 0
        self.spin_left = int(round(360 / c.sensor.turn_deg)) if c.plan.initial_spin else 0
        self.spin_reason = "START"
        self.target = None             # 当前选中的候选
        self.prev_yaw = pose[2]
        self.approach = None           # STOP 前的靠近阶段：dict(steps_left)
        self.blocked_wait = 0
        self.blocked_relaxed = False
        self.skipped_last = False
        self.exclude_yaws = []
        self.relax_until = -1          # 被人挡太久：在这一步之前放宽人的余量
        self.last_look_xy = None       # 上次低头查看的位置
        self.observe_turns = 0         # 模型要求"转过去看"时剩余的转向步数
        self.observe_action = LEFT
        self.observe_pending = False
        self.observed_last = False     # 上一次决策是"转过去看"：这次不允许再看，必须选
        self.failed = []               # 失败过的方向：dict(x, y, floor, reason)
        self.n_observes = 0
        self.stop_rejects = 0          # 模型说"到了"、距离确认没通过的次数
        self.note = None               # 下一次询问模型时附带的说明（如上次停止为何没被确认）
        self.trajectory = [(pose[0], pose[1], 0)]      # (x, y, 楼层编号)
        self.last_action, self.last_pose = None, pose
        self.last_feet = height
        self.log = {"episode_id": self.episode_id, "decisions": [], "events": [], "plan": self.plan}

    # ------------------------------------------------------------------ step
    @property
    def layer(self) -> StaticLayer:
        return self.floors[self.floor]["layer"]

    def _update_floor(self, pose, height):
        """脚下高度连续几步偏离本层参考高度超过 floor_switch_h：换到参考高度相近的那层，没有就新建一层。
        楼梯走到一半时仍留在原层（楼梯本身就画在原层的高程图里）。"""
        m = self.cfg.map
        cur = self.floors[self.floor]
        if abs(height - cur["ref"]) <= m.floor_switch_h:
            self._off_floor = 0
            if abs(height - cur["ref"]) < 0.3:                  # 站稳了：参考高度慢慢贴近真实地面
                cur["ref"] = 0.9 * cur["ref"] + 0.1 * height
            return
        self._off_floor += 1
        if self._off_floor < m.floor_switch_steps:
            return
        self._off_floor = 0
        for k, f in enumerate(self.floors):
            if abs(height - f["ref"]) <= m.floor_switch_h:
                self.floor = k
                break
        else:
            self.floors.append({"ref": height, "layer": StaticLayer(self.cfg, origin_xy=pose[:2])})
            self.floor = len(self.floors) - 1
        self.log["events"].append({"step": self.step, "event": "FLOOR", "floor": self.floor, "height": round(height, 2)})
        self.target = None
        self.exe.set_path([])
        # 新楼层的图几乎是空的：像开局一样原地转一圈看清四周，再做决策
        self.spin_left = int(round(360 / self.cfg.sensor.turn_deg))
        self.spin_reason = "FLOOR"

    def act(self, rgb, depth, pose, t: float, height: float = 0.0) -> str:
        c = self.cfg
        self.feet = height
        self._update_floor(pose, height)
        if self.step >= c.max_steps - 1:                    # 第 500 个动作必须是 STOP，否则不算成功
            return self._emit(STOP, pose, "BUDGET")

        # ---- 感知 ----
        depth_m = denormalize_depth(depth, c)
        pts = backproject(depth_m, c, self.exe.pitch)
        inst, labels = self.seg.segment(rgb, depth_m.shape)
        human_px = inst > 0
        scan = points_to_scan(pts, c, human_px)
        self.layer.update(pose, pts, height, human_px[pts.rows, pts.cols])
        self.layer.mark_footprint(pose[0], pose[1], height)
        hobs = observe_humans(pts, inst, pose, c, feet_h=height)
        self.humans.update(hobs, t, visible_fn=lambda x, y: self._visible(pose, scan, x, y))
        if self.exe.pitch_steps == 0:
            self.memory.add(self.step, pose, rgb, self.floor)

        bumped = self._check_bump(pose)
        self.trig.on_step(pose)
        self.trajectory.append((pose[0], pose[1], self.floor))

        relaxed = self.step < self.relax_until
        h_margin = c.cost.human_margin_relaxed if relaxed else c.cost.human_margin
        cm = build_costmap(self.layer, pose[:2], hobs.points_xy, self.humans.unseen_discs(t, pose[:2], height), c, h_margin)
        grid = make_plan_grid(cm, c.plan.plan_resolution)
        # 放宽只缩小"保守余量"，不关掉对走动行人的预测：预测针对的是真实会撞上的风险
        predicted = self.humans.predicted_discs(t, c.human.predict_horizon_s, height) \
            if t < c.human.motion_duration_s else []
        self._h_margin = h_margin
        ctx = dict(rgb=rgb, depth_m=depth_m, labels=labels, n_visible=len(hobs.centers), t=t)

        # ---- 开局转一圈 ----
        if self.spin_left > 0:
            self.spin_left -= 1
            if self.spin_left == 0:
                terminal = self._decide(self.spin_reason, pose, grid, ctx)
                if terminal is not None:
                    return self._emit(terminal, pose, "DECISION")
                self.prev_yaw = pose[2]
            return self._emit(LEFT, pose)

        # ---- 模型要求先转过去看一眼：转完后在新视角上重新决策 ----
        if self.observe_turns > 0:
            return self._observe_step(pose)
        if self.observe_pending:
            self.observe_pending = False
            terminal = self._decide("OBSERVE", pose, grid, ctx)
            if terminal is not None:
                return self._emit(terminal, pose, "DECISION")
            if self.observe_turns > 0:
                return self._observe_step(pose)

        # ---- STOP 前的靠近阶段 ----
        if self.approach is not None:
            return self._approach_step(pose, cm, predicted, ctx)

        # ---- 被人挡住的候选：先等人让开；等不到就放宽余量再试；还不行就换方向 ----
        if self.blocked_wait > 0:
            self.blocked_wait -= 1
            if self._replan(pose, grid):
                self.blocked_wait = 0
            elif self.blocked_wait == 0 and not self.blocked_relaxed:
                self.blocked_relaxed = True
                self.relax_until = self.step + c.cost.relax_steps
                self.blocked_wait = 2                       # 下一步按放宽后的代价图再规划一次
                self.log["events"].append({"step": self.step, "event": "RELAX"})
                return self._emit(self.exe.wait_action(), pose, "BLOCKED_WAIT")
            elif self.blocked_wait == 0:
                self.exclude_yaws.append(self.target.abs_yaw)
                self._remember_failure(self.target, "blocked_by_human")
                terminal = self._decide("PATH_INVALID", pose, grid, ctx)
                if terminal is not None:
                    return self._emit(terminal, pose, "DECISION")
            else:
                return self._emit(self.exe.wait_action(), pose, "BLOCKED_WAIT")

        if self.target is not None and self.step % c.plan.replan_every == 0:
            if not self._replan(pose, grid):
                bumped = True

        look = self._maybe_look_down(pose, cm)
        if look is not None:
            return self._emit(look, pose, "LOOK_DOWN")

        replanned = False
        for _ in range(3):                                   # 一步里最多决策三次，防止死循环
            res = self.exe.step(pose, cm, predicted, self._h_margin)
            if (res.event == "PATH_INVALID" or bumped) and self.target is not None and not replanned:
                # 路线失效先对同一目标重算：只是被家具挡了一下，不值得问模型；重算失败才算决策点
                replanned = True
                if self._replan(pose, grid):
                    bumped = False
                    continue
                self._remember_failure(self.target, "dead_end")
            if res.event == "HUMAN_BLOCKED" and not relaxed:
                # 等够了人还在：放宽人的余量，下一步按放宽后的代价图重新规划，从旁边挤过去
                self.relax_until = self.step + c.cost.relax_steps
                self.log["events"].append({"step": self.step, "event": "RELAX"})
                return self._emit(self.exe.wait_action(), pose, "WAITING")
            invalid = bumped or res.event in ("PATH_INVALID", "HUMAN_BLOCKED")
            reason = self.trig.check(grid, pose, self.step, res.event == "SEGMENT_DONE", invalid)
            bumped = False
            if reason is None and res.action is not None:
                return self._emit(res.action, pose, res.event)
            if reason is None:
                reason = "SEGMENT_DONE"
            terminal = self._decide(reason, pose, grid, ctx)
            if terminal is not None:
                return self._emit(terminal, pose, "DECISION")
            if self.approach is not None:
                return self._approach_step(pose, cm, predicted, ctx)
            if self.observe_turns > 0:
                return self._observe_step(pose)
            if self.blocked_wait > 0:
                return self._emit(self.exe.wait_action(), pose, "BLOCKED_WAIT")
        return self._emit(LEFT, pose, "NO_PROGRESS")         # 实在没路：原地转，换个视角

    # ------------------------------------------------------------------ decision
    def _decide(self, reason, pose, grid, ctx) -> Optional[str]:
        """返回 STOP 表示立刻停；否则设置新路线 / 靠近阶段 / 等待，返回 None。"""
        c = self.cfg
        field = shortest_paths(grid, pose[:2])
        same = [(x, y) for x, y, f in self.trajectory[:-8] if f == self.floor]
        visited = np.asarray(same) if same else np.zeros((0, 2))
        cands = generate_candidates(grid, field, pose, visited, c, self.exclude_yaws,
                                    ground_fn=self.layer.ground_height, feet_h=self.feet)
        self.exclude_yaws = []
        for k in cands:
            k.tried_before = self._tried_before(k)
        subgoals = self.plan["subgoals"]
        sub = subgoals[self.sub_idx] if self.sub_idx < len(subgoals) else None
        may = subgoal_may_complete(sub, ctx["labels"], ctx["n_visible"])

        recent = self.step - self.trig.st.last_call_step < c.decision.min_call_gap_steps
        if self.decision_override is not None:
            dec = self.decision_override(self, cands, dict(ctx, reason=reason, pose=pose, field=field))
        elif not cands:
            # 四周都没有能走 1 m 的方向（多半是地图还没看全）：不问模型，原地转一圈看清楚再说
            dec = Decision(None, False, False, "none", "no candidate: look around", "nocand")
        elif recent and reason not in ("START", "FLOOR", "OVERRUN", "OBSERVE"):
            # 限流：刚问过模型又遇到决策点（多是路线反复失效），沿上次方向兜底，避免连续调用
            dec = self.slow._fallback(cands, self.prev_yaw)
            dec.source = "ratelimit"
        elif not need_llm_call(reason, cands, may, self.trig.new_opening,
                               final_subgoal=self.sub_idx >= len(subgoals) - 1, skipped_last=self.skipped_last):
            viable = [k for k in cands if k.path is not None and not k.visited]
            dec = Decision(viable[0].cid, False, False, "none", "only one way", "gate")
        else:
            images, captions = build_images(ctx["rgb"], pose, cands, self.memory, c.sensor.hfov_deg, floor=self.floor)
            dec = self.slow.decide(self.instruction, self.plan, self.sub_idx, cands,
                                   self.humans.describe(pose, self.feet), self.step, self.trig.st.traveled_m,
                                   images, captions, prev_yaw=self.prev_yaw, note=self.note,
                                   allow_observe=not self.observed_last and self.n_observes < c.decision.max_observes)
            self.note = None
        if dec.observe and (self.observed_last or self.n_observes >= c.decision.max_observes):
            dec.observe = None                               # 专家/兜底也遵守"不连续看两次"和次数上限
        self.skipped_last = dec.source == "gate"
        if dec.source not in ("gate", "nocand", "ratelimit"):    # 真正做过语义决策（模型、兜底或专家）才重置兜底计时
            self.trig.mark_called(self.step)                # 兜底计时只在真正询问过模型时重置，门控跳过不算

        self.log["decisions"].append({
            "step": self.step, "reason": reason, "pose": [round(v, 3) for v in pose], "sub_idx": self.sub_idx,
            "floor": self.floor, "height": round(self.feet, 2),
            "in_wall_zone": bool(self._in_wall_zone(grid, pose)),
            "cands": [k.summary() for k in cands], "endpoints": [list(map(float, k.endpoint)) for k in cands],
            "choose": dec.choose, "subgoal_done": dec.subgoal_done, "stop": dec.stop, "observe": dec.observe,
            "goal_side": dec.goal_side, "goal_box": dec.goal_box, "source": dec.source, "why": dec.reason,
            "cached": bool(dec.source == "llm" and self.slow.client is not None and self.slow.client.last_cached)})
        if self.diag_oracle is not None:
            exp = self.diag_oracle.label(self, cands, pose)
            self.log["decisions"][-1]["diag"] = {
                "covered": self.diag_oracle.covered(cands, pose, self.feet),
                "expert_choose": exp.choose, "expert_stop": exp.stop, "expert_observe": exp.observe}

        if dec.subgoal_done and self.sub_idx < len(subgoals) - 1:
            self.sub_idx += 1
        self.trig.reset_segment(pose, self.step)

        self.observed_last = bool(dec.observe)
        if dec.stop:
            return self._start_stop(pose, field, ctx, dec.goal_side, dec.goal_box)
        if dec.observe:
            self.n_observes += 1
            n90 = int(round(90 / c.sensor.turn_deg))
            self.observe_turns = 2 * n90 if dec.observe == "behind" else n90
            self.observe_action = RIGHT if dec.observe == "right" else LEFT
            self.target = None
            self.exe.set_path([])
            return None
        if dec.choose is None or not cands:
            self.target = None
            self.exe.set_path([])
            return None
        cand = next(k for k in cands if k.cid == dec.choose)
        self.target = cand
        self.prev_yaw = cand.abs_yaw
        if cand.path is None:
            self.blocked_wait = 2 * c.plan.max_wait_steps
            self.blocked_relaxed = False
            self.exe.set_path([])
        else:
            self.exe.set_path(truncate_path(cand.path, c.decision.segment_max_m))
        return None

    def _start_stop(self, pose, field, ctx, side, box=None) -> Optional[str]:
        """执行语义策略的停止决定，不将目标点误当成可见物体表面。

        入口等目标往往位于自由空间。框内深度属于背景墙或遮挡物，不能用来
        否决 STOP，更不能据此额外前进。目标识别错误应在决策层诊断。
        """
        self.approach = None
        self.exe.set_path([])
        return self._accept_stop("STOP_MODEL")

    def _approach_step(self, pose, cm, predicted, ctx):
        """靠近阶段：走到头、被挡或超时后，重新朝目标方向量一次深度，近了才停，否则回到决策。"""
        self.approach["steps_left"] -= 1
        res = self.exe.step(pose, cm, predicted, self._h_margin)
        if res.action is not None and self.approach["steps_left"] > 0:
            return self._emit(res.action, pose, "APPROACH")
        if res.action is not None and self.exe.level_action() == res.action:
            return self._emit(res.action, pose, "APPROACH")      # 先把相机回正再量
        d = self._depth_toward(ctx["depth_m"], pose, self.approach["yaw"])
        self.approach = None
        if d <= self.cfg.decision.stop_confirm_max_m:
            return self._emit(self._accept_stop("STOP_CONFIRMED"), pose, "DECISION")
        why = "the approach was blocked" if res.action is None else "the approach ran out of steps"
        final = self._reject_stop(f"{why} and the goal still looked {d:.1f} m away")
        if final is not None:                                   # 已经坚持太多次：按模型判断停下
            return self._emit(final, pose, "DECISION")
        return self._emit(self.exe.wait_action(), pose, "STOP_REJECTED")

    def _depth_toward(self, depth_m, pose, world_yaw) -> float:
        """当前画面里，世界朝向 world_yaw 那一列附近、中间高度带的深度中位数；不在视野内返回 inf。"""
        rel = wrap(world_yaw - pose[2])
        half = math.radians(self.cfg.sensor.hfov_deg) / 2
        if abs(rel) > half * 0.9:
            return math.inf
        h, w = depth_m.shape
        focal = (w / 2) / math.tan(half)
        u = int(w / 2 - focal * math.tan(rel))
        band = depth_m[int(h * 0.35):int(h * 0.75), max(u - 8, 0):min(u + 8, w)]
        band = band[band > 0.05]
        return float(np.median(band)) if band.size else math.inf

    def _accept_stop(self, event) -> str:
        self.log["events"].append({"step": self.step, "event": event})
        return STOP

    def _reject_stop(self, why) -> Optional[str]:
        self.stop_rejects += 1
        if self.stop_rejects > self.cfg.decision.max_stop_rejects:
            return self._accept_stop("STOP_UNCONFIRMED")
        self.log["events"].append({"step": self.step, "event": "STOP_REJECTED", "why": why})
        self.note = f"A previous stop was not confirmed: {why}. Stop only if the goal is really within about 2 m."
        self.target = None
        self.exe.set_path([])
        return None

    # ------------------------------------------------------------------ helpers
    def _observe_step(self, pose):
        self.observe_turns -= 1
        if self.observe_turns == 0:
            self.observe_pending = True
        return self._emit(self.observe_action, pose, "OBSERVE")

    def _remember_failure(self, cand, reason):
        if cand is not None:
            self.failed.append({"x": cand.endpoint[0], "y": cand.endpoint[1], "floor": self.floor,
                                "reason": reason, "step": self.step})

    def _tried_before(self, cand) -> Optional[str]:
        for f in reversed(self.failed):
            if f["floor"] == self.floor and math.hypot(f["x"] - cand.endpoint[0], f["y"] - cand.endpoint[1]) < 1.0:
                return f["reason"]
        return None

    def _replan(self, pose, grid) -> bool:
        """朝当前目标重算路线；到不了返回 False。"""
        if self.target is None:
            return True
        field = shortest_paths(grid, pose[:2])
        goal = nearest_reachable(field, *self.target.endpoint)
        if goal is None:
            return False
        path = extract_path(field, goal)
        if (math.hypot(self.target.endpoint[0] - pose[0], self.target.endpoint[1] - pose[1]) >= .35
                and math.hypot(path[-1][0] - pose[0], path[-1][1] - pose[1]) < .3):
            return False  # 被挡目标吸附回脚边，不能清零等待并当作已经恢复通路。
        self.exe.set_path(truncate_path(path, self.cfg.decision.segment_max_m), new_target=False)
        return True

    def _maybe_look_down(self, pose, cm) -> Optional[str]:
        """前方 look_down_ahead_m 内的路线穿过未观测的格子：低头看一眼。
        水平相机看不到脚前 1.25 m 以内、比脚下低的地方，下楼梯和断崖都藏在这里。
        下一步执行器会自动把相机抬回水平（Executor.level_action），所以一次查看花两步。"""
        m = self.cfg.map
        if self.exe.pitch_steps != 0 or not self.exe.path:
            return None
        if self.last_look_xy is not None and \
                math.hypot(pose[0] - self.last_look_xy[0], pose[1] - self.last_look_xy[1]) < m.look_down_every_m:
            return None
        pts = np.asarray(self.exe.path)
        d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
        ahead = pts[(d > 0.3) & (d < m.look_down_ahead_m)]
        if not len(ahead):
            return None
        i, j = cm.world_to_cell(ahead[:, 0], ahead[:, 1])
        ok = cm.inside(i, j)
        if not cm.unknown[i[ok], j[ok]].any():
            return None
        self.last_look_xy = (pose[0], pose[1])
        return DOWN

    def _check_bump(self, pose) -> bool:
        """前进后几乎没动：被挡住了（多半是没看到的障碍或人）。在前方补一条障碍证据。"""
        if self.last_action != FORWARD:
            return False
        moved = math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
        p = self.cfg.plan
        if moved < p.stuck_forward_m:
            event = "BUMP"
        elif (moved < p.slide_ratio * self.cfg.sensor.step_len
              and abs(self.feet - self.last_feet) < 0.05):      # 楼梯上水平位移本来就会变短，不算
            event = "SLIDE"
        else:
            return False
        bx = self.last_pose[0] + 0.3 * math.cos(self.last_pose[2])
        by = self.last_pose[1] + 0.3 * math.sin(self.last_pose[2])
        self.layer.force_obstacle(bx, by)
        self.log["events"].append({"step": self.step, "event": event})
        return True

    @staticmethod
    def _in_wall_zone(grid, pose) -> bool:
        i, j = (int(v) for v in grid.to_cell(pose[0], pose[1]))
        return bool(grid.inside(i, j) and grid.wall_lethal[i, j])

    def _visible(self, pose, scan, x, y) -> bool:
        """这个位置此刻应该能被看到：在视场内、距离不远、射线没在它前面被挡住。"""
        f, l = world_to_agent(pose, x, y)
        dist, b = math.hypot(f, l), math.atan2(l, f)
        half = math.radians(self.cfg.sensor.hfov_deg) / 2
        if f <= 0 or abs(b) > half * 0.9 or dist > 6.0:
            return False
        k = int(np.argmin(np.abs(scan.bearings - b)))
        return bool(scan.valid[k] and scan.ranges[k] >= dist - 0.5)

    def _emit(self, action, pose, event=None):
        self.exe.note_action(action)
        if event in ("BUMP", "BLOCKED_WAIT", "NO_PROGRESS", "BUDGET", "APPROACH_DONE"):
            self.log["events"].append({"step": self.step, "event": event})
        self.last_action, self.last_pose, self.last_feet = action, pose, self.feet
        self.step += 1
        return action
