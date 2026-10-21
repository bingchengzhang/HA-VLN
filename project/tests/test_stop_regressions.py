"""STOP 不得被背景深度改写成前进，也不能在控制分支中丢失。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from havln_agent.agent import HierarchicalAgent
from havln_agent.config import AgentCfg
from havln_agent.executor import Executor
from havln_agent.slow.slow_loop import Decision
from tests.fixtures.box_scene import corridor_scene
from tests.test_agent import run_episode


def test_free_space_goal_does_not_trigger_motion_toward_background():
    cfg = AgentCfg()
    agent = object.__new__(HierarchicalAgent)
    agent.cfg, agent.exe = cfg, Executor(cfg)
    agent.exe.set_path([(0., 0.), (3., 0.)])
    agent.log, agent.step = {'events': []}, 55
    action = agent._start_stop((0., 0., 0.), None,
                               {'depth_m': np.full((256, 256), 5.427)},
                               'center', [152, 132, 208, 290])
    assert action == 'STOP'
    assert agent.approach is None and agent.exe.path == []
    assert agent.log['events'][-1]['event'] == 'STOP_MODEL'


def test_stop_at_initial_scan_completion_is_not_discarded():
    cfg = AgentCfg()
    scene = corridor_scene(cfg)
    override = lambda agent, candidates, context: Decision(None, False, True, 'none', 'done', 'test')
    agent, pose, actions = run_episode(cfg, scene, max_steps=30, override=override)
    assert actions[-1] == 'STOP'
    assert len(actions) == int(round(360 / cfg.sensor.turn_deg))


if __name__ == '__main__':
    test_free_space_goal_does_not_trigger_motion_toward_background()
    test_stop_at_initial_scan_completion_is_not_discarded()
    print('STOP regressions passed')
