"""真实验收中暴露的临界台阶和短距离脱困回归。"""
import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from havln_agent.config import AgentCfg
from havln_agent.mapping.static_layer import StaticLayer, FREE, OCCUPIED
from havln_agent.mapping.costmap import build_costmap
from havln_agent.planning.grid_search import make_plan_grid, shortest_paths
from havln_agent.planning.candidates import generate_candidates
from havln_agent.oracle import progress_target, OracleDecider


def test_boundary_stair_measurement_is_not_a_wall():
    cfg = AgentCfg()
    layer = StaticLayer(cfg)
    i, j = layer.world_to_cell(0., 0.)
    layer.ground[i, j] = -0.105
    layer.top[i, j] = 0.097
    assert layer.classify(i, i + 1, j, j + 1)[0, 0] == FREE
    layer.top[i, j] = 0.145
    assert layer.classify(i, i + 1, j, j + 1)[0, 0] == OCCUPIED


def test_step_barrier_retains_real_height_limit():
    cfg = AgentCfg()
    layer = StaticLayer(cfg)
    i, j = layer.world_to_cell(0., 0.)
    layer.ground[i, j] = 0.
    layer.ground[i + 1, j] = .202
    assert layer.classify(i, i + 2, j, j + 1)[1, 0] == FREE
    layer.ground[i + 1, j] = .25
    assert layer.classify(i, i + 2, j, j + 1)[1, 0] == OCCUPIED


def test_current_footprint_is_not_reclassified_as_a_step_barrier():
    cfg = AgentCfg()
    layer = StaticLayer(cfg)
    # 台阶边缘观测与身体脚印紧邻；高差检测不能重新封住当前身体，但外侧真实落差仍要阻挡。
    for x, h in [(-.15, .1), (.15, .4), (.2, .0)]:
        i, j = layer.world_to_cell(x, 0.)
        layer.ground[i, j] = h
    layer.mark_footprint(0., 0., .4)
    i0, j0 = layer.world_to_cell(-.4, -.4)
    result = layer.classify(i0, i0 + 16, j0, j0 + 16)
    for i in range(i0, i0 + 16):
        for j in range(j0, j0 + 16):
            x, y = layer.cell_to_world(i, j)
            if np.hypot(x, y) <= cfg.cost.agent_radius:
                assert result[i - i0, j - j0] == FREE
    i, j = layer.world_to_cell(.15, 0.)
    assert result[i - i0, j - j0] == OCCUPIED


def alcove_candidates(human_points):
    cfg = AgentCfg()
    layer = StaticLayer(cfg)
    layer.ground[:] = 0.
    for offset in np.arange(-.8, .801, layer.res):
        for x, y in [(-.8, offset), (.8, offset), (offset, -.8), (offset, .8)]:
            layer.force_obstacle(x, y)
    cm = build_costmap(layer, (0., 0.), human_points, [], cfg)
    grid = make_plan_grid(cm, cfg.plan.plan_resolution)
    field = shortest_paths(grid, (0., 0.))
    return generate_candidates(grid, field, (0., 0., 0.), np.zeros((0, 2)), cfg)


def test_narrow_alcove_has_executable_short_candidates():
    candidates = alcove_candidates(np.zeros((0, 2)))
    assert candidates, '空间允许短距离移动时不应一直原地转向'
    assert all(c.path and np.hypot(*c.path[-1]) >= .35 for c in candidates)
    assert all(c.straight_m < 1. for c in candidates)


def test_short_candidates_do_not_bypass_human_blockage():
    angles = np.linspace(0., 2 * np.pi, 48, endpoint=False)
    humans = np.column_stack((.45 * np.cos(angles), .45 * np.sin(angles)))
    candidates = alcove_candidates(humans)
    assert all(c.path is None and c.blocked_by_human for c in candidates)


def test_planner_preserves_narrow_but_safe_costmap_gap():
    cfg = AgentCfg()
    layer = StaticLayer(cfg)
    layer.ground[:] = 0.
    for y in np.arange(-10., 10.001, .05):
        if abs(y) >= .25:
            layer.force_obstacle(0., y)
    cm = build_costmap(layer, (-.5, 0.), np.zeros((0, 2)), [], cfg)
    grid = make_plan_grid(cm, cfg.plan.plan_resolution)
    field = shortest_paths(grid, (-.5, 0.))
    i, j = grid.to_cell(.5, 0.)
    assert np.isfinite(field.dist[i, j]), '规划降采样不能额外封死原代价图中的安全通路'


def test_oracle_lookahead_interpolates_sparse_route():
    target, progress = progress_target([(0., 0., 0.), (20., 0., 0.)], 4., 0., 0.)
    assert np.allclose(target, (7., 0., 0.))
    assert abs(progress - .2) < 1e-9


def test_oracle_lookahead_follows_route_corner():
    target, _ = progress_target([(0., 0., 0.), (5., 0., 0.), (5., 10., 0.)], 4., 0., 0.)
    assert np.allclose(target, (5., 2., 0.))


def test_oracle_projection_prefers_current_floor():
    route = [(0., 0., 0.), (10., 0., 0.), (10., 0., 3.), (0., 0., 3.)]
    target, _ = progress_target(route, 4., 0., 3.)
    assert np.allclose(target, (1., 0., 3.))


def test_oracle_respects_exhausted_observation_budget():
    from types import SimpleNamespace
    cfg = AgentCfg()
    agent = SimpleNamespace(feet=0., plan={'subgoals': [{}]}, sub_idx=0,
                            observed_last=False, n_observes=cfg.decision.max_observes)
    candidate = SimpleNamespace(cid=0, path=[(0., 0.), (1., 0.)], endpoint=(1., 0.))
    oracle = OracleDecider(cfg, [(0., 0., 0.), (-10., 0., 0.)], (-10., 0., 0.))
    decision = oracle.label(agent, [candidate], (0., 0., 0.))
    assert decision.observe is None and decision.choose == 0


def test_oracle_can_select_blocked_route_for_waiting():
    from types import SimpleNamespace
    cfg = AgentCfg()
    agent = SimpleNamespace(feet=0., plan={'subgoals': [{}]}, sub_idx=0,
                            observed_last=False, n_observes=cfg.decision.max_observes)
    candidate = SimpleNamespace(cid=0, path=None, endpoint=(1., 0.), blocked_by_human=True)
    oracle = OracleDecider(cfg, [(0., 0., 0.), (10., 0., 0.)], (10., 0., 0.))
    decision = oracle.label(agent, [candidate], (0., 0., 0.))
    assert decision.observe is None and decision.choose == 0
    assert candidate.path is None, '选择被挡候选只应触发等待，不能伪造可通行路线'


def test_revisited_corner_exposes_obstacle_avoiding_exit():
    cfg = AgentCfg()
    cfg.map.size_m = 12.
    layer = StaticLayer(cfg)
    ii, jj = np.mgrid[:layer.n, :layer.n]
    x, y = layer.cell_to_world(ii, jj)
    corridor = (((x >= -.3) & (x <= 1.) & (abs(y) <= .3)) |
                ((x >= .4) & (x <= 1.) & (y >= 0.) & (y <= 2.3)) |
                ((x >= -2.4) & (x <= 1.) & (y >= 1.7) & (y <= 2.3)))
    layer.ground[:] = 0.
    layer.forced[:] = ~corridor
    cm = build_costmap(layer, (0., 0.), np.zeros((0, 2)), [], cfg)
    grid = make_plan_grid(cm, cfg.plan.plan_resolution)
    field = shortest_paths(grid, (0., 0.))
    visited = np.array([(a, b) for a in np.arange(-.5, 1.51, .25)
                        for b in np.arange(-.5, 1.51, .25)])
    candidates = generate_candidates(grid, field, (0., 0., 0.), visited, cfg)
    exits = [c for c in candidates if c.path and c.endpoint[0] < -.8 and c.endpoint[1] > 1.4]
    assert exits, '已到访的局部直线不能遮蔽真正可绕行的远端出口'
    for candidate in exits:
        for px, py in candidate.path:
            i, j = cm.world_to_cell(px, py)
            assert not cm.wall_lethal[i, j], '弯曲候选仍必须通过原代价图的墙体检查'


def test_executor_follows_real_corner_without_cutting_obstacles():
    from havln_agent.executor import Executor
    data = np.load(Path(__file__).parent / 'fixtures/executor-corner.npz')

    class ObservedCostmap:
        wall_lethal = data['wall']
        human_lethal = data['human']
        dist_wall = data['dist_wall']
        agent_r = float(data['agent_r'])

        def world_to_cell(self, x, y):
            return tuple(np.floor((np.array([x, y]) - data['origin']) / float(data['res'])).astype(int))

        def inside(self, i, j):
            return 0 <= i < self.wall_lethal.shape[0] and 0 <= j < self.wall_lethal.shape[1]

    cfg, cm = AgentCfg(), ObservedCostmap()
    executor = Executor(cfg)
    executor.set_path(data['path'].tolist())
    pose = data['pose'].tolist()
    for _ in range(160):
        result = executor.step(pose, cm)
        if result.event == 'SEGMENT_DONE':
            assert np.linalg.norm(np.array(pose[:2]) - data['path'][-1]) < .3
            return
        assert result.action is not None, result.event
        if result.action == 'MOVE_FORWARD':
            nx = pose[0] + cfg.sensor.step_len * math.cos(pose[2])
            ny = pose[1] + cfg.sensor.step_len * math.sin(pose[2])
            assert not executor._segment_hits(cm, *pose[:2], nx, ny, cm.wall_lethal)
            assert not executor._segment_hits(cm, *pose[:2], nx, ny, cm.human_lethal)
            pose[:2] = [nx, ny]
        elif result.action == 'TURN_LEFT':
            pose[2] += math.radians(cfg.sensor.turn_deg)
        elif result.action == 'TURN_RIGHT':
            pose[2] -= math.radians(cfg.sensor.turn_deg)
        else:
            raise AssertionError(result.action)
    raise AssertionError('安全绕角路线没有在有限步数内执行完')


def test_replan_does_not_clear_wait_on_zero_length_snap():
    from types import SimpleNamespace
    from havln_agent.agent import HierarchicalAgent
    from havln_agent.executor import Executor
    cfg = AgentCfg()
    layer = StaticLayer(cfg)
    layer.ground[:] = 0.
    cm = build_costmap(layer, (0., 0.), np.zeros((0, 2)), [], cfg)
    cm.human_lethal[:] = True
    cm.lethal[:] = True
    grid = make_plan_grid(cm, cfg.plan.plan_resolution)
    agent = SimpleNamespace(cfg=cfg, target=SimpleNamespace(endpoint=(.55, 0.)), exe=Executor(cfg))
    assert not HierarchicalAgent._replan(agent, (0., 0., 0.), grid)
    assert not agent.exe.path, '被挡路线不能用脚边的零长度路径解除等待'


if __name__ == '__main__':
    failed = 0
    for name, test in sorted(list(globals().items())):
        if not name.startswith('test_'):
            continue
        try:
            test()
            print('PASS ', name)
        except AssertionError as exc:
            failed += 1
            print('FAIL ', name, exc)
    sys.exit(bool(failed))
