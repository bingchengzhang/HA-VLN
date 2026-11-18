"""汇总一个 run 目录：决策来源、无候选、停止方式、碰撞。只读。

用法：python scripts/diag_run.py <run_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    rows = {}
    for shard in (run / "shards").glob("*.json"):
        for row in json.loads(shard.read_text())["rows"]:
            rows[str(row["episode_id"])] = row
    for path in sorted((run / "logs").glob("*.json"), key=lambda item: item.stem):
        log = json.loads(path.read_text())
        row = rows.get(path.stem, {})
        sources = Counter(decision["source"] for decision in log["decisions"])
        events = Counter(event["event"] for event in log["events"])
        actions = Counter(row.get("actions", []))
        distance = row.get("distance_to_goal", float("nan"))
        print(
            f"ep {path.stem}: steps={len(row.get('actions', []))} dist={distance:.2f} "
            f"coll={row.get('collisions')} stop={row.get('stop_kind')} | 决策来源 {dict(sources)} | "
            f"事件 {dict(events)} | 动作 {dict(actions)}"
        )
        missing = [decision["step"] for decision in log["decisions"] if decision["source"] == "nocand"]
        if missing:
            tail = " ..." if len(missing) > 15 else ""
            print(f"    nocand 步：{missing[:15]}{tail}")
        for event in log["events"]:
            if event["event"] == "STOP_REJECTED":
                print(f"    step {event['step']} STOP_REJECTED：{event.get('why')}")


if __name__ == "__main__":
    main()
