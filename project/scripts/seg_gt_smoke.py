"""人体真值掩码冒烟测试（方案 A）：semantic 传感器 + 给每个行人物体设语义 ID。只读诊断，不进正式推理。

背景：YOLO 在 SMPL 渲染人上的召回从未测过；要测，先得有逐像素真值。
目的：验证三件事，过了才写正式的 eval_segmenter：
  1. semantic 图里能出现我们给行人设的 ID（行人确实被画进 semantic 图）
  2. semantic 与 RGB 像素对齐（叠加图上人形轮廓重合）
  3. 行人每 0.1 s 被删掉重建，新物体也能拿到 ID（转一圈期间 ID 持续出现）
做法：拦截 sim.add_object_by_handle——HASimulator/environments.py 里行人就是用它加进场景的，
      每加一个物体立刻 set_object_semantic_id(HUMAN_BASE + object_id)。不改官方代码。

  $HRUN scripts/seg_gt_smoke.py --data-root "$DATA" --source-root "$SRC" --gpu 0 \
      --seg-weights "$ZBC/weights/yolov8m-seg.pt" --out "$ZBC/workspace/diag/seg-smoke"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from havln_agent.adapters.habitat_env import make_env, EpisodeRunner
from havln_agent.rollout import load_episodes

HUMAN_BASE = 1_000_000          # 远大于 MP3D 物体实例编号，避免与场景语义冲突


def hook_human_ids(sim):
    """之后每个经 add_object_by_handle 加入的物体都带语义 ID；返回计数器方便确认钩子生效。"""
    orig = sim.add_object_by_handle
    stats = {"added": 0, "set_failed": 0}

    def wrapped(*a, **kw):
        oid = orig(*a, **kw)
        stats["added"] += 1
        try:
            sim.set_object_semantic_id(HUMAN_BASE + int(oid), oid)
        except Exception as e:                       # 记下来，不吞掉：失败就说明方案 A 走不通
            stats["set_failed"] += 1
            stats["last_error"] = repr(e)
        return oid

    sim.add_object_by_handle = wrapped
    return stats


def pick_episode(data_root, split, motion_path, max_dist=4.0):
    """选一条起点附近（max_dist 米内、同一楼层）就有行人的 episode，保证原地转一圈能看到人。
    行人位置取第 0 帧；habitat 坐标里 y 是高度，水平距离用 x、z。"""
    with open(motion_path) as f:
        motion = json.load(f)
    for ep in load_episodes(data_root, split):
        scan = Path(ep["scene_id"]).stem
        sx, sy, sz = ep["start_position"]
        for h in motion.get(scan, {}).values():
            hx, hy, hz = h["translation"][0]
            if abs(hy - sy) < 2.0 and np.hypot(hx - sx, hz - sz) < max_dist:
                return ep, scan
    raise SystemExit(f"没有找到起点 {max_dist} m 内有行人的 episode")


def overlay(rgb, mask, color):
    out = rgb[..., :3].copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--split", default="train")         # 诊断用训练集，不碰 val
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seg-weights", default=None)       # 给了就顺便看 YOLO 在同一帧上的结果
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    motion = args.data_root + "/Multi-Human-Annotations/human_motion.json"
    ep, scan = pick_episode(args.data_root, args.split, motion)
    print(f"episode {ep['episode_id']}  scan {scan}")

    env = make_env(args.data_root, args.split, [str(ep["episode_id"])], args.gpu, args.source_root,
                   extra_sensors=("SEMANTIC_SENSOR",))
    seg = None
    if args.seg_weights:
        from havln_agent.perception.human_seg import YoloSegSegmenter
        seg = YoloSegSegmenter(args.seg_weights)
    try:
        sim = env.habitat_env.sim
        stats = hook_human_ids(sim)                     # 必须在 reset 之前：reset 时第一次加行人
        runner = EpisodeRunner(env)
        obs = runner.reset()
        print("观测键：", {k: getattr(v, "shape", None) for k, v in obs.items()})
        if "semantic" not in obs:
            raise SystemExit("观测里没有 semantic：传感器没加上，检查 SEMANTIC_SENSOR 配置")
        if obs["semantic"].shape[:2] != obs["rgb"].shape[:2]:
            raise SystemExit(f"semantic {obs['semantic'].shape} 与 rgb {obs['rgb'].shape} 尺寸不一致")

        from PIL import Image
        rows, saved = [], 0
        for k in range(25):                             # 原地转一圈多一点
            sem = obs["semantic"].astype(np.int64)
            human = sem >= HUMAN_BASE
            ids = sorted(set(np.unique(sem[human]).tolist()))
            row = {"step": k, "t": round(runner.now(), 2), "human_px": int(human.sum()), "human_ids": len(ids)}
            if seg is not None:
                inst, _ = seg.segment(obs["rgb"], human.shape)
                pred = inst > 0
                row["yolo_px"] = int(pred.sum())
                row["pixel_recall"] = round(float((pred & human).sum() / max(human.sum(), 1)), 3)
            rows.append(row)
            print(row)
            if human.sum() > 0 and saved < 4:           # 存几张叠加图：红=真值，绿=YOLO
                img = overlay(obs["rgb"], human, (255, 0, 0))
                if seg is not None:
                    img = overlay(img, pred, (0, 255, 0))
                Image.fromarray(img).save(out / f"step{k:02d}.png")
                saved += 1
            obs, _, _ = runner.step("TURN_LEFT")

        print("钩子统计：", stats)
        verdict = {
            "hook_called": stats["added"] > 0,
            "set_id_ok": stats["set_failed"] == 0,
            "humans_in_semantic": any(r["human_px"] > 0 for r in rows),
            "frames_with_humans": sum(r["human_px"] > 0 for r in rows),
        }
        print("结论：", verdict)
        (out / "smoke.json").write_text(json.dumps({"episode": str(ep["episode_id"]), "scan": scan,
                                                    "stats": stats, "rows": rows, "verdict": verdict},
                                                   indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
