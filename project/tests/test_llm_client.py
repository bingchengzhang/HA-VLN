"""慢回路的 HTTP 调用测试：起一个假的 OpenAI 兼容服务，检查请求结构、图片编码、缓存和解析。

不需要真模型，验证的是 vLLM / DeepSeek 接口这一侧的代码。运行：python3 tests/test_llm_client.py
"""
import base64
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image

from havln_agent.config import AgentCfg
from havln_agent.planning.candidates import Candidate
from havln_agent.slow.llm_client import ChatClient
from havln_agent.slow.slow_loop import SlowLoop

SEEN = []


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        reply = '{"choose": 1, "observe": null, "subgoal_done": false, "stop": false, "goal_box": null, "reason": "ok"}'
        out = json.dumps({"choices": [{"message": {"content": "```json\n" + reply + "\n```"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_vlm_request_and_cache():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as cache:
        cfg = AgentCfg.from_dict({"slow": {"base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                                           "cache_dir": cache, "model": "qwen-vl"}})
        client = ChatClient(cfg.slow)
        slow = SlowLoop(cfg, client)
        cands = [Candidate(cid=k, bearing_deg=b, abs_yaw=0.0, endpoint=(2, k), straight_m=2.0,
                           path=[(0, 0), (2, k)], leads_to_unknown=np.bool_(True)) for k, b in ((0, 30), (1, 0))]
        plan = {"subgoals": [{"action": "go_straight", "landmark": "hall", "done_when": "x"}], "goal": {}}
        imgs = [np.full((224, 224, 3), 128, np.uint8), np.zeros((224, 224, 3), np.uint8)]
        caps = ["Current view.", "Earlier view."]
        d = slow.decide("walk", plan, 0, cands, [], 3, 1.0, imgs, caps)
        assert d.source == "llm" and d.choose == 1
        body = SEEN[-1]
        user = body["messages"][1]["content"]
        kinds = [part["type"] for part in user]
        assert kinds == ["text", "text", "image_url", "text", "image_url"], kinds
        raw = base64.b64decode(user[2]["image_url"]["url"].split(",", 1)[1])
        assert Image.open(io.BytesIO(raw)).size == (cfg.slow.image_side, cfg.slow.image_side)
        assert body["temperature"] == 0.0 and body["model"] == "qwen-vl"
        n = len(SEEN)
        slow.decide("walk", plan, 0, cands, [], 3, 1.0, imgs, caps)      # 同样的输入：命中缓存，不再发请求
        assert len(SEEN) == n and client.n_cache_hits == 1
    srv.shutdown()



def test_compile_uses_long_budget_and_retries_without_cache():
    """指令拆解：用 compile_max_tokens；第一次回复被截断时，第二次必须绕过缓存真正重发。"""
    from havln_agent.config import SlowCfg
    from havln_agent.slow.slow_loop import compile_instruction

    class FakeClient:
        def __init__(self):
            self.cfg, self.calls = SlowCfg(), []

        def chat(self, system, text, images=None, image_captions=None, max_tokens=None, use_cache=True):
            self.calls.append((max_tokens, use_cache))
            if len(self.calls) == 1:
                return '{"subgoals": [{"action": "exit", "landmark": "bedr'      # 被截断
            return '{"subgoals": [{"action": "exit", "landmark": "bedroom"}], "goal": {"landmark": "sofa"}}'

    c = FakeClient()
    plan = compile_instruction(c, "exit the bedroom and stop at the sofa")
    assert not plan.get("fallback"), plan
    assert c.calls == [(1024, True), (1024, False)], c.calls


if __name__ == "__main__":
    failed = 0
    for t in (test_vlm_request_and_cache, test_compile_uses_long_budget_and_retries_without_cache):
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if failed else 0)
