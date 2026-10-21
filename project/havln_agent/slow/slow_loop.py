"""慢回路：指令拆解（离线）与决策点上的选择（在线）。

在线部分只回答三件事：选几号候选、当前子目标完成没有、该不该停。
输出不合格（不是 JSON、编号不存在）重试一次；仍不行就退回"选和上次方向最接近的候选"，
保证 agent 不会因为大模型出错而卡住。
"""
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

from ..config import AgentCfg
from ..geometry import wrap
from .llm_client import ChatClient, LLMError, extract_json
from .prompts import COMPILE_SYSTEM, compile_user, DECIDE_SYSTEM, decide_user


# ---------------- 指令拆解（离线） ----------------
def fallback_plan(instruction: str) -> dict:
    """拆解失败时的兜底：整条指令当成一个子目标。"""
    return {"subgoals": [{"id": 0, "action": "go_straight", "landmark": "the goal described in the instruction",
                          "ref_human": False, "done_when": "the final location of the instruction is reached"}],
            "goal": {"landmark": "the final location", "relation": "stop at"}, "fallback": True}


def validate_plan(plan: dict) -> dict:
    sgs = plan.get("subgoals")
    if not isinstance(sgs, list) or not sgs:
        raise ValueError("没有 subgoals")
    clean = []
    for k, sg in enumerate(sgs):
        if not isinstance(sg, dict):
            raise ValueError("subgoal 不是对象")
        clean.append({"id": k, "action": str(sg.get("action", "")), "landmark": str(sg.get("landmark", "")),
                      "ref_human": bool(sg.get("ref_human", False)), "done_when": str(sg.get("done_when", ""))})
    goal = plan.get("goal") or {}
    return {"subgoals": clean, "goal": {"landmark": str(goal.get("landmark", clean[-1]["landmark"])),
                                        "relation": str(goal.get("relation", "stop at"))}}


def compile_instruction(client: ChatClient, instruction: str) -> dict:
    """第二次尝试不读缓存：第一次的回复要是被截断了，读缓存只会拿回同一份坏回复。"""
    for attempt in range(2):
        try:
            text = client.chat(COMPILE_SYSTEM, compile_user(instruction),
                               max_tokens=client.cfg.compile_max_tokens, use_cache=attempt == 0)
            return validate_plan(extract_json(text))
        except (ValueError, LLMError):
            continue
    return fallback_plan(instruction)


class PlanStore:
    """离线拆解结果：一个 jsonl 文件，每行 {"episode_id", "plan"}。"""

    def __init__(self, path: Optional[str]):
        self.plans = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.plans[str(row["episode_id"])] = row["plan"]

    def get(self, episode_id, instruction, client: Optional[ChatClient]):
        ep = str(episode_id)
        if ep in self.plans:
            return self.plans[ep]
        # 缓存里没有：在线补拆一次（大模型调用本身也有磁盘缓存）
        plan = compile_instruction(client, instruction) if client else fallback_plan(instruction)
        self.plans[ep] = plan
        return plan


# ---------------- 决策（在线） ----------------
def _as_bool(v) -> Optional[bool]:
    """严格的布尔解析：bool("false") 在 Python 里是 True，模型偶尔会输出字符串，必须单独处理。
    认不出的值返回 None（整条输出判为不合格）。"""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no", "1", "0"):
        return v.strip().lower() in ("true", "yes", "1")
    return None


@dataclass
class Decision:
    choose: Optional[int]
    subgoal_done: bool
    stop: bool
    goal_side: str         # 目标在画面的左/中/右（没有 goal_box 时的退路）
    reason: str
    source: str            # "llm" / "fallback" / "gate" / "expert" ...
    goal_box: Optional[list] = None    # 停止时目标在当前画面里的像素框 [x1, y1, x2, y2]（image_side 尺度）
    observe: Optional[str] = None      # "left" / "right" / "behind"：候选都不对，先转过去看

    def to_json(self) -> dict:
        """训练标签与模型输出用同一个格式。"""
        return {"choose": self.choose, "observe": self.observe, "subgoal_done": self.subgoal_done,
                "stop": self.stop, "goal_box": self.goal_box, "reason": self.reason}


class SlowLoop:
    def __init__(self, cfg: AgentCfg, client: Optional[ChatClient]):
        self.cfg = cfg
        self.client = client

    def build_text(self, instruction, plan, sub_idx, cands, humans, step, traveled_m, captions, with_images,
                   note=None):
        """推理和生成训练数据共用同一份 prompt，保证训练和使用时模型看到的格式一致。"""
        summaries = [c.summary() for c in cands]
        return decide_user(instruction, plan, sub_idx, summaries, humans, step, self.cfg.max_steps,
                           traveled_m, captions if with_images else [], note)

    def decide(self, instruction, plan, sub_idx, cands, humans, step, traveled_m,
               images=None, captions=None, prev_yaw=None, allow_observe=True, note=None) -> Decision:
        """allow_observe=False：上一次决策已经是"转过去看"，这次必须在候选里选（防止原地来回看）。"""
        with_images = bool(images) and self.cfg.slow.supports_images
        text = self.build_text(instruction, plan, sub_idx, cands, humans, step, traveled_m, captions, with_images,
                               note)
        valid_ids = {c.cid for c in cands}
        if self.client is not None:
            for _ in range(1 + self.cfg.slow.retries):
                try:
                    out = extract_json(self.client.chat(DECIDE_SYSTEM, text, images, captions))
                    d = self._parse(out, valid_ids, allow_observe)
                    if d is not None:
                        return d
                except (ValueError, LLMError):
                    continue
        return self._fallback(cands, prev_yaw)

    def _parse(self, out: dict, valid_ids, allow_observe=True) -> Optional[Decision]:
        stop, sub_done = _as_bool(out.get("stop", False)), _as_bool(out.get("subgoal_done", False))
        if stop is None or sub_done is None:
            return None                         # "maybe" 之类：不合格，重试
        choose = out.get("choose", None)
        if choose is not None:
            try:
                choose = int(choose)
            except (TypeError, ValueError):
                return None
            if choose not in valid_ids:
                return None
        observe = out.get("observe")
        observe = str(observe).lower() if observe else None
        if observe not in (None, "left", "right", "behind") or (observe and not allow_observe):
            observe = None
        if choose is None and not stop and not observe and valid_ids:
            return None                         # 既不选、也不停、也不看：不合格，重试
        box = self._parse_box(out.get("goal_box"))
        side = "none"
        if stop:
            side = "center"
            if box is not None:
                cx = (box[0] + box[2]) / 2 / self.cfg.slow.image_side
                side = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        return Decision(choose, sub_done, stop, side,
                        str(out.get("reason", ""))[:200], "llm", goal_box=box,
                        observe=None if (stop or choose is not None) else observe)

    def _parse_box(self, box) -> Optional[list]:
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        n = self.cfg.slow.image_side
        x1, x2 = sorted((min(max(x1, 0), n - 1), min(max(x2, 0), n - 1)))
        y1, y2 = sorted((min(max(y1, 0), n - 1), min(max(y2, 0), n - 1)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def _fallback(cands, prev_yaw) -> Decision:
        viable = [c for c in cands if c.path is not None] or cands
        if not viable:
            return Decision(None, False, False, "none", "no candidate", "fallback")
        if prev_yaw is None:
            best = min(viable, key=lambda c: abs(c.bearing_deg))
        else:
            best = min(viable, key=lambda c: abs(wrap(c.abs_yaw - prev_yaw)) + (1.0 if c.visited else 0.0))
        return Decision(best.cid, False, False, "none", "fallback: keep heading", "fallback")
