"""汇总一个 run 目录：决策来源、无候选、停止方式、碰撞。只读。用法：python diag_run.py <run_dir>"""
import json, sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
rows = {}
for f in (run / "shards").glob("*.json"):
    for r in json.loads(f.read_text())["rows"]:
        rows[str(r["episode_id"])] = r
for f in sorted((run / "logs").glob("*.json"), key=lambda p: p.stem):
    log = json.loads(f.read_text())
    r = rows.get(f.stem, {})
    src = Counter(d["source"] for d in log["decisions"])
    ev = Counter(e["event"] for e in log["events"])
    acts = Counter(r.get("actions", []))
    print(f"ep {f.stem}: steps={len(r.get('actions', []))} dist={r.get('distance_to_goal', float('nan')):.2f} "
          f"coll={r.get('collisions')} stop={r.get('stop_kind')} | 决策来源 {dict(src)} | "
          f"事件 {dict(ev)} | 动作 {dict(acts)}")
    nc = [d["step"] for d in log["decisions"] if d["source"] == "nocand"]
    if nc:
        print(f"    nocand 步：{nc[:15]}{' ...' if len(nc) > 15 else ''}")
    for e in log["events"]:
        if e["event"] == "STOP_REJECTED":
            print(f"    step {e['step']} STOP_REJECTED：{e.get('why')}")
