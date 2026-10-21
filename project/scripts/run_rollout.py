"""在官方镜像里跑智能体，导出动作序列，可选本地打分。

示例（容器内）：
  # 冒烟测试：每个 split 前 20 条，不调大模型（决策用兜底规则），本地打分
  python scripts/run_rollout.py --splits val_unseen --limit 20 --no-llm --local-metrics --output-dir /workspace/runs/smoke

  # 正式：val_seen + val_unseen 全量，调本地 vLLM
  python scripts/run_rollout.py --splits val_seen val_unseen --plans /workspace/plans/val.jsonl \
      --output-dir /workspace/runs/v1 --gpu-ids 0 1

多卡时每张卡起一个子进程，分到不同的场景分片；中断后同一命令可续跑。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_agent(args):
    from havln_agent.agent import HierarchicalAgent
    from havln_agent.config import AgentCfg
    from havln_agent.perception.human_seg import build_segmenter
    from havln_agent.slow.llm_client import ChatClient
    from havln_agent.slow.slow_loop import SlowLoop, PlanStore
    cfg = AgentCfg.from_json(args.cfg)
    seg = build_segmenter(args.segmenter, args.seg_weights, device=f"cuda:{args.gpu}", conf=cfg.human.det_conf,
                          min_pixels=cfg.human.min_mask_pixels)
    client = None if args.no_llm else ChatClient(cfg.slow)
    stats = json.loads(Path(args.path_stats).read_text()) if args.path_stats and Path(args.path_stats).exists() else None
    return HierarchicalAgent(cfg, seg, SlowLoop(cfg, client), PlanStore(args.plans), stats)


def worker(args):
    from havln_agent.rollout import run_job
    if args.oracle_decisions:
        args.local_metrics = True
    jobs = json.loads(Path(args.worker_jobs).read_text())
    out = Path(args.output_dir)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(args)
    for job in jobs:
        rows = run_job(job, agent, args, out, log_dir)
        print(f"完成 {job['key']}：{len(rows)} 条", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["val_unseen"])
    p.add_argument("--data-root", default="/data/havln2")
    p.add_argument("--source-root", default="/workspace/cma-deps/HA-VLN")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cfg", default=None, help="覆盖默认参数的 JSON 文件")
    p.add_argument("--plans", default=None, help="compile_instructions.py 生成的子目标缓存")
    p.add_argument("--path-stats", default=None, help="compile_instructions.py --fit-path-stats 的输出")
    p.add_argument("--segmenter", default="yolo", choices=["yolo", "none"])
    p.add_argument("--seg-weights", default="/weights/yolov8m-seg.pt")
    p.add_argument("--no-llm", action="store_true", help="不调大模型，决策点用兜底规则（冒烟测试用）")
    p.add_argument("--local-metrics", action="store_true")
    p.add_argument("--oracle-decisions", action="store_true",
                   help="诊断用：决策点由特权专家代替大模型，测候选机制与建图的上限。结果禁止提交")
    p.add_argument("--limit", type=int, default=None, help="每个 split 只跑前 N 条")
    p.add_argument("--episode-list", default=None, help="只跑文件里列出的 episode id（每行一个）")
    p.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    p.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--worker-jobs", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker_jobs:
        return worker(args)

    from havln_agent.rollout import load_episodes, make_jobs, summarize
    out = Path(args.output_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    if args.oracle_decisions:
        args.local_metrics = True
        (out / "ORACLE_RUN_NOT_FOR_SUBMISSION").write_text("这次运行用了特权专家做决策，只能用于诊断，禁止提交。\n")
    ids = Path(args.episode_list).read_text().split() if args.episode_list else None
    jobs = []
    for split in args.splits:
        jobs += make_jobs(load_episodes(args.data_root, split), split, ids, args.limit)
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    procs = []
    passthrough = [a for a in sys.argv[1:]]
    for k, gpu in enumerate(args.gpu_ids):
        part = jobs[k::len(args.gpu_ids)]
        jp = out / f"jobs-gpu{gpu}.json"
        jp.write_text(json.dumps(part))
        cmd = [sys.executable, __file__, *passthrough, "--gpu", str(gpu), "--worker-jobs", str(jp)]
        log = (out / f"worker-gpu{gpu}.log").open("a")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    codes = [pr.wait() for pr, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        print(f"有子进程失败（退出码 {codes}），看 {out}/worker-gpu*.log；已完成的分片会保留，重跑同一命令即可续跑")
        sys.exit(1)

    if args.local_metrics:
        for split in args.splits:
            allowed = Path(args.data_root) / f"HA-R2R-tools/collision_num_{split}.json"
            s = summarize(out, split, str(allowed))
            if s:
                print(split, json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
