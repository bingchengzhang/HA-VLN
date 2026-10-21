"""门控评估：在合成场景里走真实的门控路径，统计模型调用次数、间隔和原因。运行：python3 tests/eval_gate.py"""
import sys, math, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import Counter
from tests.test_agent import *
from tests.fixtures.box_scene import BoxScene
from havln_agent.slow.slow_loop import SlowLoop, Decision

class CountingSlow(SlowLoop):
    def __init__(s, cfg, goal, stop_r=2.0):
        super().__init__(cfg, None); s.goal=goal; s.stop_r=stop_r; s.calls=0
    def decide(s, instruction, plan, sub_idx, cands, humans, step, traveled_m, images=None, captions=None, prev_yaw=None, **kw):
        s.calls += 1
        p = s.agent.last_pose
        if math.hypot(s.goal[0]-p[0], s.goal[1]-p[1]) < s.stop_r:
            return Decision(None, False, True, "none", "", "llm")
        v=[c for c in cands if c.path is not None] or cands
        if not v: return Decision(None, False, False, "none", "", "llm")
        b=min(v,key=lambda c: math.hypot(c.endpoint[0]-s.goal[0], c.endpoint[1]-s.goal[1]))
        return Decision(b.cid, False, False, "none", "", "llm")

def run(scene, goal, steps=300, stop_r=2.0, plan=None):
    cfg=AgentCfg(); seg=FixtureSegmenter(); slow=CountingSlow(cfg, goal, stop_r)
    ag=HierarchicalAgent(cfg, seg, slow, PlanStore(None)); slow.agent=ag
    pose=(0.0,0.0,0.0); ag.reset("t","x",pose)
    if plan: ag.plan=plan
    acts=[]
    for k in range(steps):
        d,inst,rgb=scene.render(pose, ag.exe.pitch); seg.next_inst=inst
        a=ag.act(rgb,d,pose,0.1*k); acts.append(a)
        if a=="STOP": break
        pose=scene.step(pose,a)
    src=Counter(x['source'] for x in ag.log['decisions']); rs=Counter(x['reason'] for x in ag.log['decisions'])
    gaps=[b['step']-a['step'] for a,b in zip([d for d in ag.log['decisions'] if d['source']=='llm'][:-1],[d for d in ag.log['decisions'] if d['source']=='llm'][1:])]
    return pose, acts, slow.calls, src, rs, ag, scene, gaps

cfg=AgentCfg()
two=[{"action":"go_straight","landmark":"hall","done_when":"x"},{"action":"stop","landmark":"end","done_when":"y"}]
walls=[((-1,-0.8),(8,-0.8)),((-1,0.8),(8,0.8)),((8,-0.8),(8,0.8)),((-1,-0.8),(-1,0.8))]
long_walls=[((-1,-0.8),(16,-0.8)),((-1,0.8),(16,0.8)),((16,-0.8),(16,0.8)),((-1,-0.8),(-1,0.8))]
for name, scene, goal, kw in [
    ("L 形走廊，目标在拐角后", corridor_scene(cfg), (6.0,5.0), {}),
    ("L 形走廊 + 行人", corridor_scene(cfg, humans=[(3.0,-0.35)]), (6.0,5.0), {}),
    ("直走廊，目标在中段", BoxScene(cfg,walls), (5.0,0.0), dict(stop_r=1.5)),
    ("16 m 长走廊，两个子目标，目标在 12 m 处（门控可跳过前半段）", BoxScene(cfg,long_walls), (12.0,0.0), dict(stop_r=1.5, plan={"subgoals":two,"goal":{}})),
    ("死胡同，目标在墙外（到不了）", BoxScene(cfg,walls), (20.0,0.0), dict(steps=200)),
]:
    pose,acts,calls,src,rs,ag,sc,gaps=run(scene, goal, **kw)
    print(f"== {name}\n  {len(acts)} 步，结束 {acts[-1]}，离目标 {math.hypot(pose[0]-goal[0],pose[1]-goal[1]):.2f} m，碰撞 {sc.collisions}，"
          f"模型调用 {calls}，来源 {dict(src)}，原因 {dict(rs)}，两次调用间隔(步) {gaps}")
