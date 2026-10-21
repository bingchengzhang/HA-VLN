"""不用 Docker，直接从镜像仓库把官方镜像里的指定目录拷出来（在能上外网的 WSL 里跑，只用 Python 标准库）。

背景：比赛代码默认跑在官方镜像里，依赖镜像中的 /app（比赛配置、打过补丁的 habitat-lab）和
/opt/havln-challenge（评分包 havln_scoring、提交校验、任务配置）。服务器不能用 Docker，
所以把这些文件原样取出来，上传到 /mnt/data/zbc/image/ 下，用 PYTHONPATH 和配置路径接上。

做法：按 Docker Registry v2 协议匿名拉取镜像清单和每一层（层就是 tar.gz），
按顺序解出匹配的路径（后面的层覆盖前面的层，处理删除标记 .wh.），同时把镜像里所有文件的路径写进 index.txt，
方便查找 Python 包实际装在哪。

  python3 extract_image_files.py --out ./havln-image --cache ./havln-image-blobs
  # 默认取 /app、/opt/havln-challenge，以及所有 site-packages 下的 .pth / .egg-link（定位包路径用）

--cache 保存下载的层（按摘要命名，下载完校验 sha256），中断后重跑会跳过已下好的层。整个镜像可能有十几 GB。
"""
import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

IMAGE = "ghcr.io/jostarxiong/havln-challenge-2026@sha256:e1a0544f66beaf5218cc9df63da51b4a1a22a6bf0471ca0c2f75e81ee02a6556"
ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
DEFAULT_PREFIXES = ["app/", "opt/havln-challenge/"]
DEFAULT_GLOBS = ["*/site-packages/*.pth", "*/site-packages/*.egg-link"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _get(url, headers=None, timeout=120):
    """GET；层文件会 307 跳到对象存储，跳转时不能带仓库的 Authorization 头，所以手动处理跳转。"""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        return _opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return urllib.request.urlopen(e.headers["Location"], timeout=timeout)
        raise


def parse_ref(ref):
    host, rest = ref.split("/", 1)
    if "@" in rest:
        name, digest = rest.split("@", 1)
    else:
        name, digest = rest.rsplit(":", 1) if ":" in rest else (rest, "latest")
    return host, name, digest


def token(host, name):
    with _get(f"https://{host}/token?scope=repository:{name}:pull") as r:
        return json.loads(r.read())["token"]


def manifest(host, name, ref, auth):
    with _get(f"https://{host}/v2/{name}/manifests/{ref}", {"Accept": ACCEPT, **auth}) as r:
        m = json.loads(r.read())
    if "manifests" in m:                               # 多架构索引：选 linux/amd64
        pick = [d for d in m["manifests"]
                if d.get("platform", {}).get("os") == "linux" and d["platform"].get("architecture") == "amd64"]
        if not pick:
            raise SystemExit("镜像里没有 linux/amd64 版本")
        return manifest(host, name, pick[0]["digest"], auth)
    return m


def fetch_blob(host, name, digest, auth, cache: Path) -> Path:
    path = cache / digest.replace(":", "_")
    if path.exists():
        return path
    part = path.with_suffix(".partial")
    h = hashlib.sha256()
    with _get(f"https://{host}/v2/{name}/blobs/{digest}", auth, timeout=600) as r, part.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(8 << 20)
            if not buf:
                break
            f.write(buf)
            h.update(buf)
            done += len(buf)
            if total:
                print(f"\r  {digest[7:19]} {done / 2**20:8.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    if "sha256:" + h.hexdigest() != digest:
        part.unlink()
        raise SystemExit(f"层 {digest} 校验失败，重跑即可")
    os.replace(part, path)
    return path


def _clean(name):
    while name.startswith("./"):
        name = name[2:]
    p = PurePosixPath(name.lstrip("/"))
    if str(p) in ("", "."):
        return None
    if ".." in p.parts:
        return None
    return str(p)


def wanted(path, prefixes, globs):
    return any(path == p.rstrip("/") or path.startswith(p) for p in prefixes) or \
        any(fnmatch.fnmatch(path, g) for g in globs)


def _remove(target: Path):
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


def apply_layer(fileobj, out: Path, prefixes, globs, index: dict, layer_id: str):
    """把一层里匹配的文件解到 out，处理删除标记；index 记录镜像里每个路径来自哪一层。"""
    n = 0
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for m in tar:
            path = _clean(m.name)
            if not path:
                continue
            base = PurePosixPath(path).name
            if base.startswith(".wh."):                    # 删除标记
                parent = str(PurePosixPath(path).parent)
                if base == ".wh..wh..opq":                 # 整个目录被替换成新层的版本
                    for k in [k for k in index if k.startswith(parent + "/")]:
                        index.pop(k)
                    if wanted(parent + "/", prefixes, globs) and (out / parent).is_dir():
                        shutil.rmtree(out / parent)
                        (out / parent).mkdir(parents=True)
                    continue
                gone = str(PurePosixPath(parent) / base[4:])
                for k in [k for k in index if k == gone or k.startswith(gone + "/")]:
                    index.pop(k)
                if wanted(gone, prefixes, globs):
                    _remove(out / gone)
                continue
            index[path] = layer_id
            if not wanted(path, prefixes, globs):
                continue
            target = out / path
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                _remove(target)
            if m.issym():
                os.symlink(m.linkname, target)             # 原样保留；指向镜像内绝对路径的链接在本机会是断的
            elif m.islnk():
                src = out / (_clean(m.linkname) or "")
                if src.is_file():
                    shutil.copy2(src, target)
            elif m.isfile():
                with tar.extractfile(m) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.chmod(target, m.mode & 0o755 | 0o600)
                n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True, help="层文件缓存目录，支持断点续跑")
    p.add_argument("--prefix", nargs="*", default=DEFAULT_PREFIXES, help="要取出的镜像内目录（不带开头的 /）")
    p.add_argument("--glob", nargs="*", default=DEFAULT_GLOBS, help="额外按通配符取出的文件")
    args = p.parse_args()

    out, cache = Path(args.out), Path(args.cache)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} 非空；换一个目录，避免和旧结果混在一起")
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    host, name, ref = parse_ref(args.image)
    auth = {"Authorization": f"Bearer {token(host, name)}"}
    m = manifest(host, name, ref, auth)
    layers = m["layers"]
    print(f"{args.image}\n共 {len(layers)} 层，压缩后合计 {sum(l['size'] for l in layers) / 2**30:.1f} GB")
    with fetch_blob(host, name, m["config"]["digest"], auth, cache).open() as f:
        (out / "image_config.json").write_text(json.dumps(json.load(f), indent=2))
    index = {}
    for k, layer in enumerate(layers):
        blob = fetch_blob(host, name, layer["digest"], auth, cache)
        with blob.open("rb") as f:
            n = apply_layer(f, out, args.prefix, args.glob, index, layer["digest"][7:19])
        print(f"第 {k + 1}/{len(layers)} 层：取出 {n} 个文件")
    (out / "index.txt").write_text("\n".join(sorted(index)) + "\n")
    print(f"完成。镜像文件清单 {out}/index.txt（{len(index)} 条），镜像环境变量见 image_config.json 的 config.Env")


if __name__ == "__main__":
    main()
