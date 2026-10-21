"""把 run_rollout.py 的分片合并成官方格式，打包并用官方工具校验。

  python scripts/make_submission.py --run-dir /workspace/runs/v1 --splits val_seen val_unseen
  （Phase 2 用 --splits test）

检查规则与官方 export_cma_submission.py 的 check_rows 相同：
ID 覆盖完整且不重复、动作合法、长度 1–500、STOP 只能在末尾、不足 500 步必须以 STOP 结尾。
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN")


def check_rows(rows, expected):
    ids = [r["episode_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("episode id 重复")
    missing = set(expected) - set(ids)
    if missing:
        raise ValueError(f"缺少 {len(missing)} 条 episode，例如 {sorted(missing)[:5]}；先把 rollout 跑完")
    for r in rows:
        a = r["actions"]
        if (not 1 <= len(a) <= 500 or any(x not in ACTIONS for x in a)
                or (len(a) < 500 and a[-1] != "STOP") or "STOP" in a[:-1]):
            raise ValueError(f"episode {r['episode_id']} 的动作序列不合法")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["val_seen", "val_unseen"])
    p.add_argument("--data-root", default="/data/havln2")
    args = p.parse_args()
    run = Path(args.run_dir)
    if (run / "ORACLE_RUN_NOT_FOR_SUBMISSION").exists():
        raise SystemExit("这个目录是 --oracle-decisions 的诊断运行，用了特权信息，不能提交")
    names = []
    for split in args.splits:
        rows = []
        for shard in sorted((run / "shards").glob(f"{split}-*.json")):
            rows += [{"episode_id": r["episode_id"], "actions": r["actions"]}
                     for r in json.loads(shard.read_text())["rows"]]
        with gzip.open(Path(args.data_root) / f"HA-R2R/{split}/{split}.json.gz", "rt") as f:
            expected = [str(e["episode_id"]) for e in json.load(f)["episodes"]]
        check_rows(rows, expected)
        rows.sort(key=lambda r: int(r["episode_id"]))
        name = f"{split}.json"
        (run / name).write_text(json.dumps({"format_version": 1, "split": split, "episodes": rows}))
        names.append(name)
        print(f"{split}: {len(rows)} 条，格式检查通过")

    partial = run / "submission.zip.partial"
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(run / n, arcname=n)          # 不带父目录，官方会拒绝嵌套路径
    # 原生环境没有 havln-validate 命令时，用 HAVLN_VALIDATE 指定等价命令（见 docs/TRAINING.md「提交」）
    validator = os.environ.get("HAVLN_VALIDATE", "").split() or (["havln-validate"] if shutil.which("havln-validate") else [])
    if validator:
        subprocess.run([*validator, str(partial)], check=True)
    else:
        print("警告：找不到官方校验器 havln-validate，也没设置 HAVLN_VALIDATE；只做了本地格式检查，提交前务必补上官方校验")
    os.replace(partial, run / "submission.zip")
    print(f"已生成 {run / 'submission.zip'}，上传到 CodaBench")


if __name__ == "__main__":
    main()
