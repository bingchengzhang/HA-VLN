"""离线评测慢回路：把验证集的决策样本逐条发给模型服务，对比专家标签。不需要仿真器。

每次微调后先跑它，比跑完整 rollout 快得多（几百条样本几分钟），用来比较 checkpoint、调 prompt：

  python scripts/eval_decision_offline.py --data /workspace/decision_data/merged --limit 300
  python scripts/eval_decision_offline.py --data ... --cfg configs/lora.json    # 评测 LoRA 版（slow.model = nav）

指标：
  json_ok        输出能解析且合法的比例（不合格会触发重试，太低说明格式没学会）
  choose_acc     专家要"选候选"的样本里，选对编号的比例
  stop_recall    专家说该停时模型也停的比例；stop_false 是专家不让停时模型却停了的比例（错误停止）
  observe_match  专家要"转身看"时模型也要求看的比例
  box_center_err 停止样本里框中心与专家框中心的像素误差中位数（image_side 尺度）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from havln_agent.config import AgentCfg
from havln_agent.slow.llm_client import ChatClient, LLMError, extract_json
from havln_agent.slow.prompts import DECIDE_SYSTEM
from havln_agent.slow.slow_loop import SlowLoop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="write_split 的输出目录（含 decision_val_raw.json）")
    p.add_argument("--cfg", default=None)
    p.add_argument("--limit", type=int, default=300)
    args = p.parse_args()
    cfg = AgentCfg.from_json(args.cfg)
    client = ChatClient(cfg.slow)
    slow = SlowLoop(cfg, client)
    data = Path(args.data)
    rows = json.loads((data / "decision_val_raw.json").read_text())[:args.limit]

    n_ok = 0
    ch_n = ch_ok = st_n = st_ok = nst_n = nst_bad = ob_n = ob_ok = 0
    box_err = []
    for r in rows:
        imgs = [np.asarray(Image.open(data / p).convert("RGB")) for p in r["images"]]
        lab = r["label"]
        try:
            out = extract_json(client.chat(DECIDE_SYSTEM, r["text"], imgs, r["captions"]))
            d = slow._parse(out, set(r["valid_ids"]))
        except (ValueError, LLMError):
            d = None
        n_ok += d is not None
        if lab["stop"]:
            st_n += 1
            st_ok += bool(d and d.stop)
            if d and d.stop and d.goal_box and lab["goal_box"]:
                c = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                box_err.append(float(np.linalg.norm(c(d.goal_box) - c(lab["goal_box"]))))
        else:
            nst_n += 1
            nst_bad += bool(d and d.stop)
        if lab["observe"]:
            ob_n += 1
            ob_ok += bool(d and d.observe)
        elif lab["choose"] is not None:
            ch_n += 1
            ch_ok += bool(d and d.choose == lab["choose"])
    rate = lambda a, b: round(a / b, 3) if b else None
    res = {"n": len(rows), "json_ok": rate(n_ok, len(rows)), "choose_acc": rate(ch_ok, ch_n),
           "stop_recall": rate(st_ok, st_n), "stop_false": rate(nst_bad, nst_n),
           "observe_match": rate(ob_ok, ob_n),
           "box_center_err_px": round(float(np.median(box_err)), 1) if box_err else None,
           "model": cfg.slow.model}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    (data / f"offline_eval_{cfg.slow.model.replace('/', '_')}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
