"""采集数据的标签自动检查 + 人工抽查图。只读采集目录，输出写到 --out。

背景：标签错了训练再多也没用；但逐条人工看 30 道题太费时间。
做法：能用规则查的全部自动查（格式、编号合法、停止框在画面内……），人只需要扫一眼缩略图：
      每格是模型会看到的第一张图（带候选编号），停止题上画出专家的 goal_box，下方写标签。
抽样分层：停止、转身看、跨楼层候选、普通选择各取一部分，保证三类重点题都看得到。

  $HPY scripts/label_sheet.py $ZBC/workspace/decision_data/pilot --out <目录> --k 30
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

TILE, CAP, COLS = 336, 46, 5


def kind_of(lab):
    return "stop" if lab.get("stop") else "observe" if lab.get("observe") else "choose"


def check(row):
    """返回这道题的问题列表（空表示没发现问题）。"""
    bad = []
    lab = row["raw"]["label"]
    n_img = sum(m["content"].count("<image>") for m in row["messages"])
    if n_img != len(row["images"]):
        bad.append(f"<image> {n_img} 个 vs 图片 {len(row['images'])} 张")
    try:
        ans = json.loads(row["messages"][-1]["content"])
    except json.JSONDecodeError:
        return bad + ["答案不是合法 JSON"]
    if kind_of(ans) != kind_of(lab):
        bad.append("答案和原始标签类型不一致")
    k = kind_of(lab)
    if k == "choose" and lab.get("choose") not in row["raw"].get("valid_ids", []):
        bad.append(f"choose={lab.get('choose')} 不在候选编号 {row['raw'].get('valid_ids')} 里")
    if k == "stop":
        box = lab.get("goal_box")
        if box is None:
            bad.append("停止题没有 goal_box")
        else:
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= TILE and 0 <= y1 < y2 <= TILE):
                bad.append(f"goal_box {box} 不在 {TILE}×{TILE} 画面内或宽高为 0")
            elif (x2 - x1) * (y2 - y1) < 50:
                bad.append(f"goal_box {box} 太小（终点可能很远或被裁掉）")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted((data / "shards").glob("*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{data}/shards 里没有样本")
    kinds = Counter(kind_of(r["raw"]["label"]) for r in rows)
    eps = Counter(r["episode_id"] for r in rows)
    eps_stop = {r["episode_id"] for r in rows if r["raw"]["label"].get("stop")}
    problems = [(r["episode_id"], i, b) for i, r in enumerate(rows) for b in check(r)]
    floor_rows = [i for i, r in enumerate(rows) if re.search(r'"goes": "(up|down)"', r["raw"]["text"])]

    # 分层抽样：停止 / 转身看 / 跨楼层候选 / 其余选择
    rng = random.Random(a.seed)
    pools = {"stop": [i for i, r in enumerate(rows) if kind_of(r["raw"]["label"]) == "stop"],
             "observe": [i for i, r in enumerate(rows) if kind_of(r["raw"]["label"]) == "observe"],
             "floor": floor_rows}
    quota = {"stop": a.k // 3, "observe": a.k // 6, "floor": a.k // 6}
    pick = []
    for name, pool in pools.items():
        pool = [i for i in pool if i not in pick]
        pick += rng.sample(pool, min(quota[name], len(pool)))
    rest = [i for i in range(len(rows)) if i not in pick]
    pick += rng.sample(rest, min(a.k - len(pick), len(rest)))

    sheets = []
    for s in range(0, len(pick), COLS * 3):
        chunk = pick[s:s + COLS * 3]
        nrow = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * TILE, nrow * (TILE + CAP)), "white")
        d = ImageDraw.Draw(sheet)
        for k, i in enumerate(chunk):
            r = rows[i]
            lab = r["raw"]["label"]
            img = Image.open(data / r["images"][0]).convert("RGB").resize((TILE, TILE))
            x, y = (k % COLS) * TILE, (k // COLS) * (TILE + CAP)
            sheet.paste(img, (x, y))
            if lab.get("stop") and lab.get("goal_box"):
                x1, y1, x2, y2 = lab["goal_box"]
                d.rectangle([x + x1, y + y1, x + x2, y + y2], outline=(255, 0, 0), width=3)
            tag = {"stop": "STOP", "observe": f"LOOK {lab.get('observe')}",
                   "choose": f"CHOOSE {lab.get('choose')}"}[kind_of(lab)]
            extra = " floor" if i in floor_rows else ""
            d.text((x + 6, y + TILE + 4), f"#{i} {r['episode_id']} {tag}{extra}", fill=(0, 0, 0))
            d.text((x + 6, y + TILE + 22), f"imgs={len(r['images'])} sub_done={lab.get('subgoal_done')}", fill=(90, 90, 90))
        path = out / f"sheet-{s // (COLS * 3) + 1}.jpg"
        sheet.save(path, quality=85)
        sheets.append(path.name)
    (out / "samples.json").write_text(json.dumps([dict(index=i, episode_id=rows[i]["episode_id"],
                                                       label=rows[i]["raw"]["label"], text=rows[i]["raw"]["text"],
                                                       images=rows[i]["images"]) for i in pick],
                                                 ensure_ascii=False, indent=1))

    lines = [f"- 样本 {len(rows)} 道，来自 {len(eps)} 条 episode（平均每条 {len(rows) / len(eps):.1f} 道）",
             f"- 标签分布：{dict(kinds)}；含上下楼候选的题 {len(floor_rows)} 道",
             f"- 有停止题的 episode：{len(eps_stop)}/{len(eps)}"
             + ("（偏少：专家很少走到能停的位置，先查原因再扩量）" if len(eps_stop) < 0.6 * len(eps) else ""),
             f"- 自动检查发现问题 {len(problems)} 处" + ("：" if problems else "")]
    lines += [f"  - ep{e} 第 {i} 题：{b}" for e, i, b in problems[:20]]
    lines += [f"- 抽查图：{', '.join(sheets)}（红框 = 专家 goal_box）；每张图对应的完整题目文字见 samples.json",
              "- 人工只需看：停止题的红框是否落在指令终点附近、LOOK 题是不是确实看不到正确方向、"
              "CHOOSE 的编号方向是否大致朝参考路线"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
