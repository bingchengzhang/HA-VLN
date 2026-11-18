"""门控评估：在合成场景里走真实的门控路径，统计模型调用次数、间隔和原因。

运行：python3 tests/eval_gate.py
"""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from havln_agent.agent import HierarchicalAgent
from havln_agent.config import AgentCfg
from havln_agent.slow.slow_loop import Decision, PlanStore, SlowLoop

from tests.fixtures.box_scene import BoxScene, FixtureSegmenter, corridor_scene


class CountingSlow(SlowLoop):
    def __init__(self, cfg, goal, stop_r=2.0):
        super().__init__(cfg, None)
        self.goal = goal
        self.stop_r = stop_r
        self.calls = 0

    def decide(
        self,
        instruction,
        plan,
        sub_idx,
        cands,
        humans,
        step,
        traveled_m,
        images=None,
        captions=None,
        prev_yaw=None,
        **kw,
    ):
        self.calls += 1
        pose = self.agent.last_pose
        if math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1]) < self.stop_r:
            return Decision(None, False, True, "none", "", "llm")
        viable = [c for c in cands if c.path is not None] or cands
        if not viable:
            return Decision(None, False, False, "none", "", "llm")
        best = min(
            viable, key=lambda c: math.hypot(c.endpoint[0] - self.goal[0], c.endpoint[1] - self.goal[1])
        )
        return Decision(best.cid, False, False, "none", "", "llm")


def run(scene, goal, steps=300, stop_r=2.0, plan=None):
    cfg = AgentCfg()
    segmenter = FixtureSegmenter()
    slow = CountingSlow(cfg, goal, stop_r)
    agent = HierarchicalAgent(cfg, segmenter, slow, PlanStore(None))
    slow.agent = agent
    pose = (0.0, 0.0, 0.0)
    agent.reset("t", "x", pose)
    if plan:
        agent.plan = plan
    actions = []
    for k in range(steps):
        depth, _inst, rgb = scene.render(pose, agent.exe.pitch)
        segmenter.next_inst = _inst
        action = agent.act(rgb, depth, pose, 0.1 * k)
        actions.append(action)
        if action == "STOP":
            break
        pose = scene.step(pose, action)
    sources = Counter(item["source"] for item in agent.log["decisions"])
    reasons = Counter(item["reason"] for item in agent.log["decisions"])
    llm_steps = [item for item in agent.log["decisions"] if item["source"] == "llm"]
    gaps = [b["step"] - a["step"] for a, b in zip(llm_steps, llm_steps[1:])]
    return pose, actions, slow.calls, sources, reasons, agent, scene, gaps


def main():
    cfg = AgentCfg()
    two = [
        {"action": "go_straight", "landmark": "hall", "done_when": "x"},
        {"action": "stop", "landmark": "end", "done_when": "y"},
    ]
    walls = [((-1, -0.8), (8, -0.8)), ((-1, 0.8), (8, 0.8)), ((8, -0.8), (8, 0.8)), ((-1, -0.8), (-1, 0.8))]
    long_walls = [
        ((-1, -0.8), (16, -0.8)),
        ((-1, 0.8), (16, 0.8)),
        ((16, -0.8), (16, 0.8)),
        ((-1, -0.8), (-1, 0.8)),
    ]
    cases = [
        ("L 形走廊，目标在拐角后", corridor_scene(cfg), (6.0, 5.0), {}),
        ("L 形走廊 + 行人", corridor_scene(cfg, humans=[(3.0, -0.35)]), (6.0, 5.0), {}),
        ("直走廊，目标在中段", BoxScene(cfg, walls), (5.0, 0.0), dict(stop_r=1.5)),
        (
            "16 m 长走廊，两个子目标，目标在 12 m 处（门控可跳过前半段）",
            BoxScene(cfg, long_walls),
            (12.0, 0.0),
            dict(stop_r=1.5, plan={"subgoals": two, "goal": {}}),
        ),
        ("死胡同，目标在墙外（到不了）", BoxScene(cfg, walls), (20.0, 0.0), dict(steps=200)),
    ]
    for name, scene, goal, kw in cases:
        pose, actions, calls, sources, reasons, _agent, sc, gaps = run(scene, goal, **kw)
        dist = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        print(
            f"== {name}\n  {len(actions)} 步，结束 {actions[-1]}，离目标 {dist:.2f} m，碰撞 {sc.collisions}，"
            f"模型调用 {calls}，来源 {dict(sources)}，原因 {dict(reasons)}，两次调用间隔(步) {gaps}"
        )


if __name__ == "__main__":
    main()
