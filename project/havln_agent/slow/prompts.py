"""慢回路的两类 prompt。

  COMPILE：离线把整条指令拆成有序子目标，每条附可检验的完成条件（提交前批量跑，结果缓存）
  DECIDE ：在决策点做选择题——选哪个候选、当前子目标完成没有、该不该停

HA-R2R 指令是英文，prompt 也用英文，模型理解更稳。输出一律是 JSON，字段在下面写死。
"""
import json

COMPILE_SYSTEM = """You convert indoor navigation instructions into an ordered list of subgoals for a robot.
The instructions describe a route through a house and often mention people and what they are doing.
Rules:
- Keep the original order. One subgoal per distinct movement or landmark.
- "landmark" is the object, room or person that marks the subgoal, in a few words.
- "ref_human" is true only if the landmark is a person or group of people.
- "done_when" is a short, visually checkable condition, e.g. "the bathroom door is directly ahead within 2 m".
- The last subgoal is where the robot must stop. Put its landmark and spatial relation in "goal".
- Do not invent landmarks that are not in the instruction.
Reply with JSON only."""

COMPILE_SCHEMA = {
    "subgoals": [{"id": 0, "action": "exit|enter|pass|turn_left|turn_right|go_straight|go_up|go_down|stop",
                  "landmark": "str", "ref_human": False, "done_when": "str"}],
    "goal": {"landmark": "str", "relation": "str"},
}


def compile_user(instruction: str) -> str:
    return ("Instruction:\n" + instruction.strip() +
            "\n\nReturn JSON with this shape:\n" + json.dumps(COMPILE_SCHEMA, ensure_ascii=False))


DECIDE_SYSTEM = """You are the planner of a robot that follows a navigation instruction inside a house with people.
A geometric module has already found the directions the robot can physically take and checked them for walls and people.
Your job is only to pick the direction that follows the instruction, and to judge progress.
Rules:
- Choose by meaning (the instruction and what you see), not by which path is shortest.
- Prefer directions with "visited": false unless the instruction requires going back.
- "tried_before" tells why a direction failed earlier. "blocked_by_human" may be free again; "dead_end" will not.
- "blocked_by_human": true means a person currently stands in that way. Choose it only if the instruction clearly
  requires that way (for example it says to pass that person); the robot will then wait for space.
- "crosses_human_zone": true means the path passes close to a person. That is allowed but less preferred.
- "goes": "up" / "down" means the path climbs or descends (stairs or a ramp); "unknown" means its end is not seen yet.
  Use it for instructions such as "go up the stairs" or "walk downstairs".
- Set "subgoal_done" true only if the current subgoal's done_when condition is clearly satisfied now.
- Set "stop" true only when the final goal is visible in the current view and close (within about 2 m).
  Then give "goal_box": [x1, y1, x2, y2], the pixel box of the goal location in the current view (Image 0),
  using the image's own pixel coordinates (origin top-left).
- If none of the candidates matches the instruction, do not guess: set "choose": null and "observe" to
  "left", "right" or "behind" to turn and look in that direction first.
Reply with JSON only: {"choose": <candidate id or null>, "observe": <"left"|"right"|"behind"|null>,
"subgoal_done": <bool>, "stop": <bool>, "goal_box": <[x1, y1, x2, y2] or null>, "reason": "<one short sentence>"}"""


def decide_user(instruction: str, plan: dict, sub_idx: int, cands: list, humans: list,
                step: int, max_steps: int, traveled_m: float, image_notes: list, note: str = None) -> str:
    subgoals = plan.get("subgoals", [])
    lines = [
        "Instruction:", instruction.strip(), "",
        "Subgoal plan:",
    ]
    for k, sg in enumerate(subgoals):
        mark = "  <-- current" if k == sub_idx else ("  (done)" if k < sub_idx else "")
        lines.append(f"  {k}. {sg.get('action', '')} {sg.get('landmark', '')} | done when: {sg.get('done_when', '')}{mark}")
    goal = plan.get("goal", {})
    lines += [f"Final goal: {goal.get('relation', '')} {goal.get('landmark', '')}", "",
              f"Progress: step {step} of {max_steps}, walked {traveled_m:.1f} m so far.", ""]
    if note:
        lines += ["Note: " + note, ""]
    lines += [
              "Candidate directions (bearing: positive = left, 0 = straight ahead):"]
    for c in cands:
        lines.append("  " + json.dumps(c, ensure_ascii=False))
    lines.append("")
    lines.append("People the robot knows about:" if humans else "People the robot knows about: none")
    for h in humans:
        lines.append("  " + json.dumps(h, ensure_ascii=False))
    if image_notes:
        lines += ["", "Images attached:"] + ["  " + n for n in image_notes]
    return "\n".join(lines)
