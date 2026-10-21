"""智能体流程测试：在合成的 L 形走廊里跑完整的快回路 + 决策层（不需要 Habitat、不调大模型）。

检查的是代码逻辑是否自洽：坐标约定、反投影、建图、候选生成、执行器、决策流程、STOP。
运行：python3 tests/test_agent.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from havln_agent.agent import HierarchicalAgent
from havln_agent.config import AgentCfg
from havln_agent.geometry import habitat_to_map, yaw_to_habitat_quat_xyzw, bearing_to
from havln_agent.slow.slow_loop import SlowLoop, PlanStore, Decision
from havln_agent.planning.candidates import Candidate
from tests.fixtures.box_scene import BoxScene, corridor_scene, FixtureSegmenter

GOAL = (6.0, 5.0)


def goal_seeking_override(agent, cands, ctx):
    """测试用"大模型"：选终点离目标最近的候选；离目标 2 m 内就停。"""
    pose = ctx["pose"]
    d = math.hypot(GOAL[0] - pose[0], GOAL[1] - pose[1])
    if d < 2.0:
        return Decision(None, False, True, "none", "goal", "test")
    viable = [c for c in cands if c.path is not None] or cands
    if not viable:
        return Decision(None, False, False, "none", "none", "test")
    best = min(viable, key=lambda c: math.hypot(c.endpoint[0] - GOAL[0], c.endpoint[1] - GOAL[1]))
    return Decision(best.cid, False, False, "none", "closest", "test")


def run_episode(cfg, scene, max_steps=300, override=None, start=(0.0, 0.0, 0.0)):
    seg = FixtureSegmenter()
    agent = HierarchicalAgent(cfg, seg, SlowLoop(cfg, None), PlanStore(None),
                              decision_override=override or goal_seeking_override)
    pose = start
    agent.reset("t0", "walk down the corridor, turn left and stop at the end", pose, scene.feet(pose))
    actions = []
    for k in range(max_steps):
        depth, inst, rgb = scene.render(pose, agent.exe.pitch)
        seg.next_inst = inst
        a = agent.act(rgb, depth, pose, t=0.1 * k, height=scene.feet(pose))
        actions.append(a)
        if a == "STOP":
            break
        pose = scene.step(pose, a)
    return agent, pose, actions


def test_geometry_conventions():
    """habitat 四元数 ↔ 地图 yaw；TURN_LEFT 使 yaw 增加。"""
    for yaw in (0.0, 0.7, -2.0, 3.0):
        q = yaw_to_habitat_quat_xyzw(yaw)
        (x, y, yaw2), _ = habitat_to_map([1.0, 0.2, -3.0], q)
        assert abs(math.remainder(yaw2 - yaw, 2 * math.pi)) < 1e-6
        assert (x, y) == (1.0, 3.0)


def test_corridor_reaches_goal_without_collision():
    cfg = AgentCfg()
    scene = corridor_scene(cfg)
    agent, pose, actions = run_episode(cfg, scene)
    d = math.hypot(pose[0] - GOAL[0], pose[1] - GOAL[1])
    assert actions[-1] == "STOP", f"没有停下，最后位置 {pose}"
    assert d < 3.0, f"停在离目标 {d:.2f} m 处"
    assert scene.collisions == 0, f"碰撞 {scene.collisions} 次"
    assert len(agent.log["decisions"]) >= 2
    print(f"    {len(actions)} 步，离目标 {d:.2f} m，决策 {len(agent.log['decisions'])} 次")


def test_escape_when_starting_inside_wall_margin():
    """起点离墙 0.15 m，落在墙的膨胀禁区（0.22 m）里：服务器 val_unseen 第 1 条就卡在这种位置原地转了 67 步。
    要求：决策点仍然有候选，agent 能先离开禁区再到达目标，不出现长串 NO_PROGRESS。"""
    cfg = AgentCfg()
    scene = corridor_scene(cfg)
    agent, pose, actions = run_episode(cfg, scene, start=(0.0, -0.65, 0.0))
    nocand = sum(d["source"] == "nocand" for d in agent.log["decisions"])
    stuck = sum(e["event"] == "NO_PROGRESS" for e in agent.log["events"])
    d = math.hypot(pose[0] - GOAL[0], pose[1] - GOAL[1])
    assert stuck <= 3, f"NO_PROGRESS {stuck} 次，nocand {nocand} 次：没能离开墙边禁区"
    assert actions[-1] == "STOP" and d < 3.0, f"没到目标：最后位置 {pose[:2]}"
    assert scene.collisions == 0, f"碰撞 {scene.collisions} 次"
    print(f"    {len(actions)} 步，nocand {nocand} 次，NO_PROGRESS {stuck} 次")


def test_human_in_corridor_is_avoided():
    """走廊中央偏一侧站着一个人：要么绕过去，要么等，但不能撞。"""
    cfg = AgentCfg()
    scene = corridor_scene(cfg, humans=[(3.0, -0.35)])
    agent, pose, actions = run_episode(cfg, scene)
    assert scene.collisions == 0, f"碰撞 {scene.collisions} 次"
    assert len(agent.humans.tracks) >= 1, "没有跟踪到行人"
    d = math.hypot(pose[0] - GOAL[0], pose[1] - GOAL[1])
    assert actions[-1] == "STOP" and d < 3.0, f"没有绕过行人到达目标：最后位置 {pose[:2]}"
    print(f"    {len(actions)} 步，最后位置 ({pose[0]:.1f}, {pose[1]:.1f})，跟踪到 {len(agent.humans.tracks)} 人")


def test_climb_stairs_to_upper_floor():
    """走廊尽头上楼梯到 1.8 m 高的二层平台，停在平台中段：要能走上去、换到二层的地图、不碰撞。"""
    from tests.fixtures.box_scene import stairs_scene
    cfg = AgentCfg()
    scene = stairs_scene(cfg)
    goal = (6.0, 0.0)

    def go(agent, cands, ctx):
        pose = ctx["pose"]
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < 1.5 and scene.feet(pose) > 1.0:
            return Decision(None, False, True, "none", "goal", "test")
        viable = [c for c in cands if c.path is not None] or cands
        if not viable:
            return Decision(None, False, False, "none", "none", "test")
        best = min(viable, key=lambda c: math.hypot(c.endpoint[0] - goal[0], c.endpoint[1] - goal[1]))
        return Decision(best.cid, False, False, "none", "closest", "test")

    agent, pose, actions = run_episode(cfg, scene, override=go)
    assert scene.collisions == 0, f"碰撞 {scene.collisions} 次"
    assert actions[-1] == "STOP" and scene.feet(pose) > 1.7, f"没上到二层：位置 {pose[:2]}，脚下 {scene.feet(pose):.2f}"
    assert len(agent.floors) == 2, f"应建了两层的图，实际 {len(agent.floors)}"
    ups = [c for d in agent.log["decisions"] for c in d["cands"] if c.get("goes") == "up"]
    assert ups, "候选里没有标出上楼方向"
    print(f"    {len(actions)} 步，脚下 {scene.feet(pose):.2f} m，楼层切换 "
          f"{[e for e in agent.log['events'] if e['event'] == 'FLOOR']}，低头 {actions.count('LOOK_DOWN')} 次")


def test_descend_stairs():
    """从 1.8 m 高的平台沿楼梯走下去，到楼下走廊的目标：下行台阶不能被当成断崖，也不能碰撞。"""
    from tests.fixtures.box_scene import BoxScene
    cfg = AgentCfg()
    walls = [((-1.0, -0.8), (7.0, -0.8)), ((-1.0, 0.8), (7.0, 0.8)),
             ((-1.0, -0.8), (-1.0, 0.8)), ((7.0, -0.8), (7.0, 0.8))]
    blocks = [(4.0, -0.8, 7.0, 0.8, 0.0, 1.8)] + \
             [(4.0 - (k + 1) * 0.28, -0.8, 4.0, 0.8, 0.0, 1.8 - (k + 1) * 0.18) for k in range(9)]
    scene = BoxScene(cfg, walls, blocks=blocks)
    goal = (0.5, 0.0)

    def go(agent, cands, ctx):
        pose = ctx["pose"]
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < 1.5 and scene.feet(pose) < 0.1:
            return Decision(None, False, True, "none", "goal", "test")
        viable = [c for c in cands if c.path is not None] or cands
        if not viable:
            return Decision(None, False, False, "none", "none", "test")
        best = min(viable, key=lambda c: math.hypot(c.endpoint[0] - goal[0], c.endpoint[1] - goal[1]))
        return Decision(best.cid, False, False, "none", "closest", "test")

    seg = FixtureSegmenter()
    agent = HierarchicalAgent(cfg, seg, SlowLoop(cfg, None), PlanStore(None), decision_override=go)
    pose = (6.0, 0.0, math.pi)
    agent.reset("t1", "walk down the stairs", pose, scene.feet(pose))
    actions = []
    for k in range(300):
        depth, inst, rgb = scene.render(pose, agent.exe.pitch)
        seg.next_inst = inst
        a = agent.act(rgb, depth, pose, t=0.1 * k, height=scene.feet(pose))
        actions.append(a)
        if a == "STOP":
            break
        pose = scene.step(pose, a)
    assert scene.collisions == 0, f"碰撞 {scene.collisions} 次"
    assert actions[-1] == "STOP" and scene.feet(pose) < 0.1, f"没下到楼下：位置 {pose[:2]}，脚下 {scene.feet(pose):.2f}"
    print(f"    {len(actions)} 步，低头 {actions.count('LOOK_DOWN')} 次，楼层 {len(agent.floors)} 张图")


def test_slow_loop_parse_and_fallback():
    """大模型输出不合格时重试，再不行退回"沿原方向"。"""
    cfg = AgentCfg()

    class Bad:
        n_calls = 0

        def chat(self, *a, **k):
            Bad.n_calls += 1
            return "I think candidate 7 is best"        # 没有 JSON

    class Good:
        def chat(self, *a, **k):
            return 'Sure. {"choose": 1, "subgoal_done": true, "stop": false, "goal_side": "none", "reason": "x"}'

    cands = [Candidate(cid=0, bearing_deg=40, abs_yaw=0.7, endpoint=(1, 1), straight_m=2, path=[(0, 0), (1, 1)]),
             Candidate(cid=1, bearing_deg=0, abs_yaw=0.0, endpoint=(2, 0), straight_m=2, path=[(0, 0), (2, 0)])]
    plan = {"subgoals": [{"action": "go_straight", "landmark": "hall", "done_when": "x"}], "goal": {}}
    d = SlowLoop(cfg, Good()).decide("go", plan, 0, cands, [], 5, 1.0)
    assert d.source == "llm" and d.choose == 1 and d.subgoal_done
    d = SlowLoop(cfg, Bad()).decide("go", plan, 0, cands, [], 5, 1.0, prev_yaw=0.0)
    assert d.source == "fallback" and d.choose == 1 and Bad.n_calls == 1 + cfg.slow.retries


def test_observe_and_goal_box_parsing():
    """observe 与 goal_box 的解析；上一次已经"看过"时不允许再看。"""
    cfg = AgentCfg()
    cands = [Candidate(cid=0, bearing_deg=0, abs_yaw=0.0, endpoint=(2, 0), straight_m=2, path=[(0, 0), (2, 0)])]
    plan = {"subgoals": [{"action": "go_straight", "landmark": "hall", "done_when": "x"}], "goal": {}}

    class Reply:
        def __init__(self, text):
            self.text = text

        def chat(self, *a, **k):
            return self.text

    look = Reply('{"choose": null, "observe": "left", "subgoal_done": false, "stop": false, "goal_box": null}')
    d = SlowLoop(cfg, look).decide("go", plan, 0, cands, [], 5, 1.0)
    assert d.observe == "left" and d.choose is None and d.source == "llm"
    d = SlowLoop(cfg, look).decide("go", plan, 0, cands, [], 5, 1.0, prev_yaw=0.0, allow_observe=False)
    assert d.observe is None and d.source == "fallback" and d.choose == 0
    stop = Reply('{"choose": null, "stop": true, "goal_box": [250, 100, 320, 300], "subgoal_done": true}')
    d = SlowLoop(cfg, stop).decide("go", plan, 0, cands, [], 5, 1.0)
    assert d.stop and d.goal_box == [250, 100, 320, 300] and d.goal_side == "right"


def test_goal_box_depth_and_projection():
    """专家把终点投影成框，执行器从框里量出的深度和方向与真值一致。"""
    from havln_agent.executor import box_depth_bearing
    from havln_agent.oracle import goal_box
    cfg = AgentCfg()
    scene = corridor_scene(cfg)
    pose = (0.0, 0.0, 0.0)
    depth, _, _ = scene.render(pose)
    d_m = depth[..., 0] * cfg.sensor.max_depth
    goal = (5.2, 0.4, 0.0)                     # L 形走廊拐角处的墙前，略偏左
    box = goal_box(cfg, pose, 0.0, goal, cfg.slow.image_side)
    assert box is not None
    dist, bearing = box_depth_bearing(d_m, box, cfg.slow.image_side, cfg.sensor.hfov_deg)
    assert abs(math.degrees(bearing) - math.degrees(math.atan2(0.4, 5.2))) < 3.0, math.degrees(bearing)
    assert 4.5 < dist < 7.0, dist              # 框里看到的是拐角处的墙和远处的地面


def test_observe_turns_then_decides():
    """模型说"往左看"：执行 90° 左转（6 步）后，在新视角上重新决策，且这次不能再要求看。"""
    cfg = AgentCfg()
    scene = corridor_scene(cfg)
    calls = []

    def override(agent, cands, ctx):
        calls.append((ctx["reason"], agent.observed_last))
        if len(calls) == 1:
            return Decision(None, False, False, "none", "look", "test", observe="left")
        return goal_seeking_override(agent, cands, ctx)

    agent, pose, actions = run_episode(cfg, scene, max_steps=60, override=override)
    spin = int(round(360 / cfg.sensor.turn_deg))
    turns = actions[spin:spin + 6]
    assert turns == ["TURN_LEFT"] * 6, actions[spin:spin + 8]
    assert calls[1] == ("OBSERVE", True), calls[:3]


def test_strict_bool_parsing():
    """模型输出字符串 "false" 时不能被当成 True；认不出的值整条判为不合格。"""
    cfg = AgentCfg()
    slow = SlowLoop(cfg, None)
    d = slow._parse({"choose": 0, "stop": "false", "subgoal_done": "true"}, {0})
    assert d is not None and d.stop is False and d.subgoal_done is True
    assert slow._parse({"choose": 0, "stop": "maybe"}, {0}) is None


def test_model_stop_does_not_move_toward_a_distant_background():
    """长厅的背景很远不代表停止位置很远；执行层不能替策略发明前进动作。"""
    cfg = AgentCfg()
    hall = [((-1.0, -5.0), (40.0, -5.0)), ((-1.0, 5.0), (40.0, 5.0)),
            ((-1.0, -5.0), (-1.0, 5.0)), ((40.0, -5.0), (40.0, 5.0))]
    scene = BoxScene(cfg, hall)
    def always_stop(agent, cands, ctx):
        return Decision(None, False, True, "center", "insist", "test")

    agent, pose, actions = run_episode(cfg, scene, max_steps=200, override=always_stop)
    ev = [e["event"] for e in agent.log["events"] if e["event"].startswith("STOP")]
    assert ev == ["STOP_MODEL"], ev
    assert actions[-1] == "STOP" and actions.count("MOVE_FORWARD") == 0
    assert math.hypot(pose[0], pose[1]) < 1e-6
    assert scene.collisions == 0


def test_view_memory_is_floor_aware():
    """上下楼后，同一平面位置的旧帧属于另一层，不能拿来当候选方向的画面。"""
    from havln_agent.slow.view_memory import ViewMemory
    m = ViewMemory()
    img = np.zeros((8, 8, 3), np.uint8)
    m.add(0, (1.0, 1.0, 0.0), img, floor=0)
    assert m.find(1.0, 1.0, 0.0, floor=1) is None
    assert m.find(1.0, 1.0, 0.0, floor=0) == 0


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
