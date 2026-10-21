"""离线把指令拆成子目标清单（提交前跑一次，结果缓存）。不需要 Habitat，宿主机或容器里都能跑。

  python scripts/compile_instructions.py --splits val_seen val_unseen --out /workspace/plans/val.jsonl
  python scripts/compile_instructions.py --fit-path-stats --out /workspace/plans/path_stats.json   # 只读 train

拆解：多线程并发调大模型，已完成的 episode 跳过，可断点续跑。
--fit-path-stats：在训练集上拟合"参考路径长度 ≈ a + b × 指令词数"，供智能体判断是否走过头。
只用训练集拟合，val/test 不参与（主办方规定 val/test 不能用于训练）。
"""
import argparse
import gzip
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from havln_agent.config import AgentCfg
from havln_agent.slow.llm_client import ChatClient
from havln_agent.slow.slow_loop import compile_instruction


def load(data_root, split):
    with gzip.open(Path(data_root) / f"HA-R2R/{split}/{split}.json.gz", "rt") as f:
        return json.load(f)["episodes"]


def fit_path_stats(data_root, out):
    xs, ys = [], []
    for e in load(data_root, "train"):
        ref = e.get("reference_path") or []
        if len(ref) < 2:
            continue
        length = sum(math.dist(a, b) for a, b in zip(ref[:-1], ref[1:]))
        xs.append(len(e["instruction"]["instruction_text"].split()))
        ys.append(length)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9)
    a = my - b * mx
    stats = {"a": a, "b": b, "n": n, "mean_len": my, "note": "path_len ≈ a + b * n_words，只用 train 拟合"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/havln2")
    p.add_argument("--splits", nargs="+", default=["val_seen", "val_unseen"])
    p.add_argument("--out", required=True)
    p.add_argument("--cfg", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fit-path-stats", action="store_true")
    p.add_argument("--episode-list", default=None,
                   help="只拆这些 episode（每行一个 id，例如 gen_decision_data.py --list-episodes 的输出）")
    p.add_argument("--limit", type=int, default=None, help="每个 split 只拆按 id 排序后的前 N 条（与 run_rollout --limit 对应）")
    args = p.parse_args()

    if args.fit_path_stats:
        return fit_path_stats(args.data_root, args.out)

    cfg = AgentCfg.from_json(args.cfg)
    client = ChatClient(cfg.slow)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        last = {r["episode_id"]: r for r in rows}                       # 同一条拆过多次，以最后一次为准
        done = {k for k, r in last.items() if not r["plan"].get("fallback")}
    keep = set(Path(args.episode_list).read_text().split()) if args.episode_list else None
    todo = []
    for s in args.splits:
        eps = sorted(load(args.data_root, s), key=lambda e: int(e["episode_id"]))
        if keep is not None:
            eps = [e for e in eps if str(e["episode_id"]) in keep]
        if args.limit:
            eps = eps[:args.limit]
        todo += [e for e in eps if str(e["episode_id"]) not in done]
    print(f"已完成 {len(done)}，待拆解 {len(todo)}")
    lock = threading.Lock()

    def one(e):
        plan = compile_instruction(client, e["instruction"]["instruction_text"])
        with lock, out.open("a") as f:
            f.write(json.dumps({"episode_id": str(e["episode_id"]), "plan": plan}, ensure_ascii=False) + "\n")
        return plan.get("fallback", False)

    with ThreadPoolExecutor(args.workers) as pool:
        n_fb = sum(pool.map(one, todo))
    print(f"完成。兜底（拆解失败）{n_fb} 条；新调用 {client.n_calls} 次，缓存命中 {client.n_cache_hits} 次")


if __name__ == "__main__":
    main()
