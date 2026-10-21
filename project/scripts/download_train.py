"""下载 HA-R2R 训练集。官方工具包只下 val 两个 split（下载清单里没有 train），训练数据要单独拉。

  python scripts/download_train.py --data-root /data/havln2
  HF_ENDPOINT=https://hf-mirror.com python scripts/download_train.py   # 国内镜像

文件版本钉在官方工具包使用的同一个 Hugging Face 提交上，下完用 gzip + json 校验能打开。
"""
import argparse
import gzip
import json
import os
import shutil
import urllib.request
from pathlib import Path

REPO = "fly1113/HA-VLN"
REVISION = "c5f1cec167964bda3ba343acc3a18e7aef8540dd"   # 与 toolkit scripts/resources.json 的 hf_revision 一致
FILES = ["HA-R2R/train/train.json.gz"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/havln2")
    p.add_argument("--revision", default=REVISION)
    args = p.parse_args()
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for rel in FILES:
        dst = Path(args.data_root) / rel
        if dst.exists():
            print("已存在", dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base}/datasets/{REPO}/resolve/{args.revision}/{rel}"
        print("下载", url)
        tmp = dst.with_name(dst.name + ".partial")
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        with gzip.open(tmp, "rt") as f:
            n = len(json.load(f)["episodes"])
        os.replace(tmp, dst)
        print(f"完成 {dst}：{n} 条 episode")


if __name__ == "__main__":
    main()
