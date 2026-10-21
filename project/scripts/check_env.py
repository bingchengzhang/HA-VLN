"""配好环境后先跑这个：确认能建环境、打印实际传感器参数，并核对 config.py 里标了 VERIFY 的值。

  python scripts/check_env.py --split val_unseen

会输出：传感器尺寸/视场角/深度范围/相机高度、导航网格最大台阶与坡度、动作集、是否允许滑动、跨楼层 episode 比例，
以及第一个 episode 走 5 步时的位姿变化（用来确认坐标约定：TURN_LEFT 后 yaw 应增加 15°）。
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from havln_agent.adapters.habitat_env import make_env, sensor_report, EpisodeRunner
from havln_agent.config import AgentCfg
from havln_agent.rollout import load_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val_unseen")
    p.add_argument("--data-root", default="/data/havln2")
    p.add_argument("--source-root", default="/workspace/cma-deps/HA-VLN")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    ep = load_episodes(args.data_root, args.split)[0]
    env = make_env(args.data_root, args.split, [str(ep["episode_id"])], args.gpu, args.source_root, with_metrics=True)
    try:
        rep = sensor_report(env)
        print(json.dumps(rep, indent=2, default=str))
        cfg = AgentCfg()
        # 行人加入后，HA-VLN 用 NavMeshSettings.set_defaults() 重算导航网格（只改了半径和身高），
        # 所以默认的最大台阶和最大坡度就是仿真器判断"能不能走"的标准，建图参数要与之一致
        import habitat_sim
        nm = habitat_sim.nav.NavMeshSettings()
        nm.set_defaults()
        rep["NAVMESH_MAX_CLIMB"] = getattr(nm, "agent_max_climb", None)
        rep["NAVMESH_MAX_SLOPE"] = getattr(nm, "agent_max_slope", None)
        d = rep.get("DEPTH_SENSOR", {})
        checks = {
            "深度宽": (d.get("WIDTH"), cfg.sensor.img_width),
            "视场角": (d.get("HFOV"), cfg.sensor.hfov_deg),
            "最大深度": (d.get("MAX_DEPTH"), cfg.sensor.max_depth),
            "深度归一化": (d.get("NORMALIZE_DEPTH"), cfg.sensor.depth_normalized),
            "相机高度": ((d.get("POSITION") or [None, None])[1], cfg.sensor.camera_height),
            "俯仰步长": (rep.get("TILT_ANGLE"), cfg.sensor.tilt_deg),
            "最大台阶": (rep.get("NAVMESH_MAX_CLIMB"), cfg.map.max_step_h),
        }
        print(f"  导航网格最大坡度 {rep.get('NAVMESH_MAX_SLOPE')}°（建图跨缝推断用 {cfg.map.gap_slope_deg}°，应不大于它）")
        print("\n核对 config.py：")
        for k, (actual, ours) in checks.items():
            flag = "未读到，需人工核实" if actual is None else "OK" if str(actual) == str(ours) or (
                isinstance(actual, (int, float)) and abs(float(actual) - float(ours)) < 1e-6) else "不一致，改 config.py"
            print(f"  {k:6s} 镜像={actual}  我们={ours}  {flag}")

        runner = EpisodeRunner(env)
        obs = runner.reset()
        print("\n指令：", EpisodeRunner.instruction(obs, env.current_episode)[:120], "...")
        print("观测：", {k: getattr(v, "shape", type(v).__name__) for k, v in obs.items()})
        p0 = runner.pose()
        for a in ["TURN_LEFT", "MOVE_FORWARD", "LOOK_UP", "LOOK_DOWN", "TURN_RIGHT"]:
            obs, done, info = runner.step(a)
            p1 = runner.pose()
            print(f"  {a:12s} 位姿 ({p1[0]:.2f}, {p1[1]:.2f}, {math.degrees(p1[2]):.1f}°)  "
                  f"碰撞计数 {(info.get('collisions_detail') or {}).get('count')}")
        print(f"  起点 yaw {math.degrees(p0[2]):.1f}°；TURN_LEFT 之后应 +15°，MOVE_FORWARD 应沿新朝向前进约 0.25 m")
    finally:
        env.close()

    # 跨楼层的 episode 占多少：参考路径高度跨度超过 1 m 视为跨楼层
    print("\n跨楼层比例：")
    for split in ("train", "val_seen", "val_unseen"):
        try:
            eps = load_episodes(args.data_root, split)
        except FileNotFoundError:
            continue
        multi = sum(max(p[1] for p in e["reference_path"]) - min(p[1] for p in e["reference_path"]) > 1.0
                    for e in eps)
        print(f"  {split}: {multi}/{len(eps)} = {multi / len(eps):.1%}")


if __name__ == "__main__":
    main()
