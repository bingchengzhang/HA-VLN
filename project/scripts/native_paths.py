"""在拷出来的镜像文件里找出比赛运行时需要的 Python 包，输出一行 PYTHONPATH（冒号分隔）。

找的包：打过补丁的 habitat（版本 0.1.7）、habitat_extensions、vlnce_baselines、HASimulator、havln_scoring，
以及镜像 site-packages 里 .pth 文件指向的目录（镜像里的 /app、/opt 改写成本机的拷贝位置）。
native_env.sh 会自动调用；单独运行可以看它找到了什么：

  $HPY scripts/native_paths.py --root /mnt/data/zbc/image --explain
"""
import argparse
import sys
from pathlib import Path

PACKAGES = ["habitat", "habitat_extensions", "vlnce_baselines", "HASimulator", "havln_scoring"]
SKIP = {"site-packages", "dist-packages"}          # 镜像自带 python 的第三方包不拿：版本可能与本机环境冲突


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--explain", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        return
    found = {}
    for pkg in PACKAGES:
        hits = []
        for init in root.rglob(f"{pkg}/__init__.py"):
            parent = init.parent.parent
            if SKIP & set(parent.parts) or "test" in str(parent).lower():
                continue
            hits.append(parent)
        hits.sort(key=lambda x: len(x.parts))
        found[pkg] = hits
    paths = []
    for pkg in PACKAGES:
        if found[pkg]:
            paths.append(found[pkg][0])
    for pth in root.rglob("*.pth"):
        for line in pth.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith(("/app", "/opt/havln-challenge")):
                local = root / line.lstrip("/")
                if local.is_dir():
                    paths.append(local)
    uniq = []
    for x in paths:
        if str(x) not in uniq:
            uniq.append(str(x))
    if args.explain:
        for pkg in PACKAGES:
            print(f"{pkg:20s} {[str(h) for h in found[pkg]] or '没找到'}", file=sys.stderr)
    print(":".join(uniq))


if __name__ == "__main__":
    main()
