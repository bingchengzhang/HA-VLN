"""训练数据生成测试：在合成走廊里用专家驾驶，检查样本格式能被 LLaMA-Factory 直接读取。

检查：<image> 数量与图片数一致、图片是推理时的分辨率、标签是合法 JSON 且能被推理端的解析器读回、
按场景划分的 dataset_info.json 字段齐全。运行：python3 tests/test_data_gen.py
"""
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from PIL import Image

from havln_agent.agent import HierarchicalAgent
from havln_agent.config import AgentCfg
from havln_agent.oracle import OracleDecider
from havln_agent.slow.llm_client import extract_json
from havln_agent.slow.slow_loop import SlowLoop, PlanStore
from gen_decision_data import Recorder, write_split
from tests.fixtures.box_scene import corridor_scene, FixtureSegmenter


def test_expert_data_roundtrip():
    cfg = AgentCfg()
    scene = corridor_scene(cfg)
    ref = [(x * 0.5, 0.0, 0.0) for x in range(12)] + [(6.0, 0.5 + y * 0.5, 0.0) for y in range(9)]
    goal = (6.0, 5.0, 0.0)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r0"
        rec = Recorder(cfg, out, perturb=0.3, dagger=False, rng=random.Random(0))
        seg = FixtureSegmenter()
        agent = HierarchicalAgent(cfg, seg, SlowLoop(cfg, None), PlanStore(None), decision_override=rec)
        rows_all = []
        for ep in range(2):
            rec.rows = []
            rec.new_episode(OracleDecider(cfg, ref, goal), f"ep{ep}", f"train-scan{ep}")
            pose = (0.0, 0.0, 0.0)
            agent.reset(f"ep{ep}", "walk down the corridor, turn left and stop at the end", pose, 0.0)
            for k in range(250):
                depth, inst, rgb = scene.render(pose, agent.exe.pitch)
                seg.next_inst = inst
                a = agent.act(rgb, depth, pose, 0.1 * k, 0.0)
                if a == "STOP":
                    break
                pose = scene.step(pose, a)
            (out / "shards").mkdir(parents=True, exist_ok=True)
            (out / "shards" / f"train-scan{ep}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rec.rows))
            rows_all += rec.rows
        assert a == "STOP" and math.hypot(pose[0] - goal[0], pose[1] - goal[1]) < 3.0, pose
        assert len(rows_all) >= 4
        kinds = set()
        for r in rows_all:
            user = r["messages"][1]["content"]
            assert user.count("<image>") == len(r["images"]) >= 1
            for p in r["images"]:
                assert Image.open(out / p).size == (cfg.slow.image_side, cfg.slow.image_side)
            ans = extract_json(r["messages"][2]["content"])
            dec = SlowLoop(cfg, None)._parse(ans, set(range(10)))
            assert dec is not None, ans
            kinds.add("stop" if dec.stop else "observe" if dec.observe else "choose")
        assert "stop" in kinds and "choose" in kinds, kinds
        merged = Path(d) / "merged"
        write_split([out], merged)
        info = json.loads((merged / "dataset_info.json").read_text())
        assert set(info) == {"decision_train", "decision_val"}
        for name in info:
            for r in json.loads((merged / f"{name}.json").read_text()):
                assert all((merged / p).exists() for p in r["images"])
        # DAgger 轮：沿用第 0 轮的验证场景与验证题，验证场景里的新样本不进训练
        held = json.loads((merged / "heldout_scans.json").read_text())
        val0 = json.loads((merged / "decision_val.json").read_text())
        merged1 = Path(d) / "merged1"
        write_split([out, out], merged1, reference=merged)
        assert json.loads((merged1 / "heldout_scans.json").read_text()) == held
        assert json.loads((merged1 / "decision_val.json").read_text()) == val0
        tr1 = json.loads((merged1 / "decision_train.json").read_text())
        tr0 = json.loads((merged / "decision_train.json").read_text())
        assert len(tr1) == 2 * len(tr0)
        raw = json.loads((merged1 / "decision_val_raw.json").read_text())
        assert raw and {"text", "captions", "label", "valid_ids", "images"} <= set(raw[0])
        print(f"    {len(rows_all)} 个样本，标签类型 {sorted(kinds)}")


if __name__ == "__main__":
    try:
        test_expert_data_roundtrip()
        print("PASS  test_expert_data_roundtrip")
    except AssertionError as e:
        print(f"FAIL  test_expert_data_roundtrip: {e}")
        sys.exit(1)
