"""阶段脚本：每个阶段一条命令，自己选卡、起停模型服务、做检查、写报告。

为什么有它：之前每一关都由执行者临时写编排脚本，又长又容易错（端口、卡号、服务启停、失败判断）。
现在这些都固定在这里，执行者只需要 source native_env.sh 后跑一条命令，再把 report.md 原样贴回来。

  GPU=none source scripts/native_env.sh          # 卡由本脚本自己挑，这里不设
  $HPY scripts/stage.py smoke                    # 不调模型的 20 条冒烟
  $HPY scripts/stage.py paired --n 20            # 原版模型 vs 专家，各 20 条（G7）
  $HPY scripts/stage.py collect --n 20 --tag pilot      # 试采 + 标签自动检查 + 抽查图
  $HPY scripts/stage.py collect --n 500 --tag r0        # 正式采集 + 划分
  $HPY scripts/stage.py offline --run r0 --model base   # 离线考试
  $HPY scripts/stage.py train --run r0                  # 试训 20 步 → 正式训练 → 合并
  $HPY scripts/stage.py offline --run r0 --model nav
  $HPY scripts/stage.py closed --run r0 --model nav --n 20 --compare <paired 报告里的 base 目录>

规则（写死在代码里，不靠执行者记）：
  - 只用 ALLOWED_GPUS（默认 5,6,7）里此刻没有任何计算进程的卡；不够就退出，绝不动别人的进程
  - 模型服务只绑 127.0.0.1、端口随机；结束时只停自己起的进程组
  - 每次运行一个新目录 $ZBC/workspace/stages/<阶段>-<tag>-<时间>，不覆盖旧结果
  - 失败就停：报告里写明停在哪一步、原因、日志位置
"""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

E = os.environ
for k in ("ZBC", "PROJECT", "HPY", "TPY", "TCLI", "DATA", "SRC"):
    if k not in E:
        sys.exit(f"缺环境变量 {k}：先 GPU=none source scripts/native_env.sh")
ZBC, PROJECT, DATA, SRC = Path(E["ZBC"]), Path(E["PROJECT"]), E["DATA"], E["SRC"]
HPY, TPY, TCLI = E["HPY"], E["TPY"], E["TCLI"]
HPATH = E.get("HAVLN_PYTHONPATH", "")
ALLOWED = [g.strip() for g in E.get("ALLOWED_GPUS", "5,6,7").split(",") if g.strip()]
SEG = str(ZBC / "weights/yolov8m-seg.pt")
PLANS = ZBC / "workspace/plans"


class StageError(Exception):
    pass


# ---------------------------------------------------------------- 基础设施
class Stage:
    def __init__(self, name, tag):
        stamp = time.strftime("%m%d-%H%M%S")
        self.dir = ZBC / "workspace/stages" / f"{name}-{tag}-{stamp}"
        self.dir.mkdir(parents=True)
        self.lines = [f"# {name}（{tag}）", "", f"- 目录：`{self.dir}`", f"- 开始：{time.strftime('%F %T')}"]
        self.status("STARTED")

    def note(self, text=""):
        self.lines.append(text)
        print(text, flush=True)

    def status(self, stage, **kw):
        (self.dir / "status.json").write_text(json.dumps(dict(stage=stage, time=time.strftime("%F %T"), **kw),
                                                         indent=2, ensure_ascii=False))

    def run(self, name, cmd, gpu=None, habitat=False, timeout=None, check=True):
        """跑一个子命令，输出写到 logs/<name>.log；失败抛 StageError 并带上日志尾巴。"""
        env = dict(E, LLM_API_KEY="EMPTY", LLAMAFACTORY_VERBOSITY="warning", TOKENIZERS_PARALLELISM="false")
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        if habitat:
            env["PYTHONPATH"] = HPATH
        log = self.dir / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        self.status("RUNNING", step=name, log=str(log))
        print(f"→ {name}（卡 {gpu}）：{' '.join(map(str, cmd))}", flush=True)
        with log.open("w") as f:
            p = subprocess.Popen([str(c) for c in cmd], cwd=PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(p)
                raise StageError(f"{name} 超时（{timeout}s），日志 {log}")
        if check and code != 0:
            raise StageError(f"{name} 退出码 {code}，日志 {log}\n```\n{_tail(log)}\n```")
        return log

    def start_bg(self, name, cmd, gpu, habitat=False):
        """后台跑（和别的子命令并行），返回 (进程, 日志)。"""
        env = dict(E, LLM_API_KEY="EMPTY", CUDA_VISIBLE_DEVICES=str(gpu))
        if habitat:
            env["PYTHONPATH"] = HPATH
        log = self.dir / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        print(f"→ {name}（卡 {gpu}，后台）", flush=True)
        f = log.open("w")
        p = subprocess.Popen([str(c) for c in cmd], cwd=PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        return p, log

    def wait_bg(self, name, p, log, timeout=None):
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(p)
            raise StageError(f"{name} 超时，日志 {log}")
        if code != 0:
            raise StageError(f"{name} 退出码 {code}，日志 {log}\n```\n{_tail(log)}\n```")

    def finish(self, ok, msg=""):
        self.lines += ["", f"## 结论：{'完成' if ok else '失败，已停止'}", msg, "", f"- 结束：{time.strftime('%F %T')}"]
        (self.dir / "report.md").write_text("\n".join(self.lines) + "\n")
        self.status("DONE" if ok else "FAILED", message=msg)
        print(f"\n报告：{self.dir / 'report.md'}")


def _tail(path, n=30):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _kill_group(p):
    """只停自己起的进程组（Popen 时 start_new_session=True）。"""
    if p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def pick_gpus(k):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuid = {r.split(",")[0].strip(): r.split(",")[1].strip() for r in rows.splitlines() if r.strip()}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    busy = {a.strip() for a in apps.splitlines() if a.strip()}
    free = [g for g in ALLOWED if g in uuid and uuid[g] not in busy]
    if len(free) < k:
        raise StageError(f"需要 {k} 张空闲卡，允许的 {ALLOWED} 里只有 {free} 空闲。没有动任何别人的进程。")
    return free[:k]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def ensure_configs(run, dataset=None):
    """train-configs/<run> 不存在就生成一次（按 2 卡）。已存在就沿用，保证同一轮配置不变。"""
    d = ZBC / "workspace/train-configs" / run
    if not d.exists():
        dataset = dataset or ZBC / "workspace/decision_data" / f"{run}-split"
        subprocess.run([TPY, "scripts/make_train_configs.py", "--run", run, "--root", ZBC,
                        "--dataset", dataset, "--num-gpus", "2"], cwd=PROJECT, check=True)
    return d


@contextmanager
def model_service(st, run, model, gpu):
    """起本轮 base/nav 模型的 OpenAI 兼容服务；yield 一份指向它的智能体配置路径。"""
    cfgdir = ensure_configs(run)
    port = free_port()
    agent = json.loads((cfgdir / f"agent-{model}.json").read_text())
    agent["slow"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    agent_path = st.dir / f"agent-{model}.json"
    agent_path.write_text(json.dumps(agent, indent=2))
    env = dict(E, CUDA_VISIBLE_DEVICES=str(gpu), API_HOST="127.0.0.1", API_PORT=str(port),
               API_MODEL_NAME=agent["slow"]["model"], API_KEY="EMPTY", LLAMAFACTORY_VERBOSITY="warning")
    log = st.dir / "logs" / f"api-{model}.log"
    log.parent.mkdir(exist_ok=True)
    f = log.open("w")
    p = subprocess.Popen([TCLI, "api", str(cfgdir / f"api-{model}.yaml")], cwd=PROJECT, env=env,
                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    (st.dir / "api.pid").write_text(f"{p.pid}\n")
    st.note(f"- 模型服务：{agent['slow']['model']}，卡 {gpu}，端口 {port}，PID {p.pid}")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while True:
            if p.poll() is not None:
                raise StageError(f"模型服务退出，日志 {log}\n```\n{_tail(log)}\n```")
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                                 headers={"Authorization": "Bearer EMPTY"})
                with opener.open(request, timeout=3) as r:
                    if json.load(r).get("data"):
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise StageError(f"模型服务 15 分钟没起来，日志 {log}")
            time.sleep(3)
        yield agent_path
    finally:
        _kill_group(p)
        f.close()
        st.note("- 模型服务已停（只停了自己起的进程）")


def path_stats():
    out = PLANS / "path-stats.json"
    if not out.exists():
        PLANS.mkdir(parents=True, exist_ok=True)
        subprocess.run([HPY, "scripts/compile_instructions.py", "--data-root", DATA, "--fit-path-stats",
                        "--out", out], cwd=PROJECT, check=True, env=dict(E, PYTHONPATH=HPATH))
    return out


def compile_plans(st, agent_cfg, gpu, out, splits, limit=None, episode_list=None):
    """拆指令；拆完检查目标 episode 里还有没有兜底，有就失败（不拿兜底计划冒充模型）。"""
    cmd = [HPY, "scripts/compile_instructions.py", "--data-root", DATA, "--splits", *splits,
           "--workers", "1", "--cfg", agent_cfg, "--out", out]
    if limit:
        cmd += ["--limit", limit]
    if episode_list:
        cmd += ["--episode-list", episode_list]
    st.run("compile-" + "-".join(splits), cmd, gpu=gpu, habitat=True, timeout=6 * 3600)
    last = {}
    for line in Path(out).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["episode_id"]] = r
    want = None
    if episode_list:
        want = set(Path(episode_list).read_text().split())
    fb = [k for k, r in last.items() if r["plan"].get("fallback") and (want is None or k in want)]
    st.note(f"- 指令拆解：{out}，兜底 {len(fb)} 条")
    if fb:
        raise StageError(f"仍有 {len(fb)} 条拆解失败（如 {fb[:5]}），看 logs/compile-*.log 和模型服务日志")


def summarize_run(st, run_dir, title):
    """把一个 rollout 目录的指标和逐条诊断写进报告。"""
    m = json.loads((Path(run_dir) / "metrics_val_unseen.json").read_text())
    diag = subprocess.run([HPY, "scripts/diag_run.py", run_dir], cwd=PROJECT, capture_output=True, text=True).stdout
    (st.dir / f"diag-{title}.txt").write_text(diag)
    return m


def metrics_table(cols):
    keys = ["n", "SR", "NE", "CR", "Score", "reach"]
    dkeys = ["candidate_coverage", "choice_agreement", "false_stop_rate", "wrong_floor_rate", "stop_rejected_per_ep"]
    names = list(cols)
    out = ["| 指标 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    for k in keys:
        out.append(f"| {k} | " + " | ".join(fmt(cols[n].get(k)) for n in names) + " |")
    for k in dkeys:
        out.append(f"| {k} | " + " | ".join(fmt(cols[n].get("diagnostics", {}).get(k)) for n in names) + " |")
    out.append("| stop_kinds | " + " | ".join(str(cols[n].get("diagnostics", {}).get("stop_kinds")) for n in names) + " |")
    out.append("| failures | " + " | ".join(str(cols[n].get("failures")) for n in names) + " |")
    out.append("| 每条模型调用 | " + " | ".join(fmt(cols[n].get("llm_decisions_per_ep")) for n in names) + " |")
    return out


def rollout_cmd(out, n, cfg=None, plans=None, extra=()):
    cmd = [HPY, "scripts/run_rollout.py", "--data-root", DATA, "--source-root", SRC, "--splits", "val_unseen",
           "--limit", n, "--gpu-ids", "0", "--segmenter", "yolo", "--seg-weights", SEG,
           "--local-metrics", "--path-stats", path_stats(), "--output-dir", out]
    if cfg:
        cmd += ["--cfg", cfg]
    if plans:
        cmd += ["--plans", plans]
    return cmd + list(extra)


# ---------------------------------------------------------------- 各阶段
def stage_smoke(st, a):
    (g,) = pick_gpus(1)
    out = st.dir / "run-nollm"
    st.run("rollout-nollm", rollout_cmd(out, a.n, extra=["--no-llm"]), gpu=g, habitat=True, timeout=4 * 3600)
    m = summarize_run(st, out, "nollm")
    st.note("\n## 不调模型的冒烟（分数无参考意义，只看链路和行为）\n")
    st.lines += metrics_table({"no-llm": m})
    st.note(f"\n逐条诊断见 diag-nollm.txt。重点看：NO_PROGRESS、SLIDE/BUMP 次数，collision_steps 是否集中。")


def stage_paired(st, a):
    g_llm, g_sim = pick_gpus(2)
    plans = PLANS / "val_unseen.jsonl"
    base_out, orc_out = st.dir / "run-base", st.dir / "run-oracle"
    with model_service(st, "r0", "base", g_llm) as cfg:
        compile_plans(st, cfg, g_sim, plans, ["val_unseen"], limit=a.n)
        # 专家不需要模型服务，放在服务那张卡上和原版并行跑，省一半时间
        p, log = st.start_bg("rollout-oracle", rollout_cmd(orc_out, a.n, plans=plans,
                                                           extra=["--no-llm", "--oracle-decisions"]), g_llm, habitat=True)
        st.run("rollout-base", rollout_cmd(base_out, a.n, cfg=cfg, plans=plans), gpu=g_sim, habitat=True,
               timeout=8 * 3600)
        st.wait_bg("rollout-oracle", p, log, timeout=4 * 3600)
    mb, mo = summarize_run(st, base_out, "base"), summarize_run(st, orc_out, "oracle")
    st.note(f"\n## 原版 vs 专家（val_unseen 前 {a.n} 条，同一份 plans）\n")
    st.lines += metrics_table({"原版 Qwen": mb, "专家（诊断，禁止提交）": mo})
    cov = mo.get("diagnostics", {}).get("candidate_coverage", 0)
    gap = mo["SR"] - mb["SR"]
    if mo["SR"] < 0.3 or cov < 0.6:
        hint = "专家本身就差：瓶颈在建图/候选/执行，先修快回路，训练救不了。"
    elif gap >= 0.15:
        hint = "专家明显好于原版：瓶颈在慢回路判断，值得采数据训练。"
    else:
        hint = "两者接近：训练收益可能有限，先看失败归因再决定。"
    st.note(f"\n自动判断（仅供参考，以你的确认为准）：{hint}")
    st.note(f"\n原版目录 `{base_out}`（closed 阶段用 --compare 指向它）。逐条诊断见 diag-base.txt / diag-oracle.txt。")


def stage_collect(st, a):
    g_llm, g_sim = pick_gpus(2)
    data = ZBC / "workspace/decision_data" / a.tag
    if data.exists():
        raise StageError(f"{data} 已存在：换一个 --tag，不和旧数据混")
    ids = st.dir / "train-ids.txt"
    plans = PLANS / "train.jsonl"
    st.run("list-episodes", [HPY, "scripts/gen_decision_data.py", "--data-root", DATA, "--limit", a.n,
                             "--seed", a.seed, "--list-episodes", "--out", ids], habitat=True)
    with model_service(st, "r0", "base", g_llm) as cfg:
        compile_plans(st, cfg, g_sim, plans, ["train"], episode_list=ids)
    # 专家驾驶不调模型：服务已停，采集只占一张卡
    st.run("collect", [HPY, "scripts/gen_decision_data.py", "--data-root", DATA, "--source-root", SRC,
                       "--plans", plans, "--cfg", cfg, "--seg-weights", SEG, "--gpu", "0",
                       "--limit", a.n, "--perturb", a.perturb, "--seed", a.seed, "--out", data],
           gpu=g_sim, habitat=True, timeout=48 * 3600)
    st.run("label-check", [HPY, "scripts/label_sheet.py", data, "--out", st.dir / "labels", "--k", "30"])
    st.lines += ["", "## 标签检查", "", (st.dir / "labels/summary.md").read_text()]
    split = ZBC / "workspace/decision_data" / f"{a.tag}-split"
    log = st.run("merge", [HPY, "scripts/gen_decision_data.py", "--merge", data, "--out", split], check=False)
    if (split / "manifest.json").exists():
        st.note(f"\n划分完成：`{split}`\n```\n{(split / 'manifest.json').read_text()}\n```")
    else:
        st.note(f"\n没有划分（试采场景太少时正常）：\n```\n{_tail(log, 5)}\n```")
    st.note(f"\n抽查图在 `{st.dir / 'labels'}`，把 sheet-*.jpg 拷回来看一眼即可。")


def stage_offline(st, a):
    (g,) = pick_gpus(1)
    split = ZBC / "workspace/decision_data" / f"{a.data}-split"
    with model_service(st, a.run, a.model, g) as cfg:
        st.run("offline-eval", [HPY, "scripts/eval_decision_offline.py", "--data", split, "--cfg", cfg,
                                "--limit", "1000000"], habitat=True, timeout=12 * 3600)
    name = json.loads(Path(cfg).read_text())["slow"]["model"]
    res = (split / f"offline_eval_{name}.json").read_text()
    st.note(f"\n## 离线考试：{name}\n```\n{res}\n```")


def validate_training_output(directory, minimum_steps):
    """先核验真实训练状态，防止退出码正常但没有有效更新时继续合并。"""
    directory = Path(directory)
    state_path = directory / "trainer_state.json"
    if not state_path.is_file():
        raise StageError(f"训练缺少状态文件：{state_path}")
    state = json.loads(state_path.read_text())
    steps = state.get("global_step", 0)
    planned_steps = state.get("max_steps", 0)
    history = state.get("log_history", [])
    losses = [r["loss"] for r in history if "loss" in r]
    eval_losses = [r["eval_loss"] for r in history if "eval_loss" in r]
    gradients = [r["grad_norm"] for r in history if r.get("grad_norm") is not None]
    finite = lambda values: all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    if (steps < minimum_steps or (planned_steps > 0 and steps < planned_steps)
            or not losses or not finite(losses + eval_losses)
            or any(v < 0 for v in losses + eval_losses) or not any(v > 0 for v in losses)):
        raise StageError(f"训练状态无效：{directory}，steps={steps}，loss={losses[-5:]}")
    if not finite(gradients) or (gradients and not any(v > 0 for v in gradients)):
        raise StageError(f"训练梯度无效：{directory}，grad_norm={gradients[-5:]}")
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise StageError(f"训练缺少非空适配器权重：{weights}")
    summary = dict(global_step=steps, loss_records=len(losses), loss_first=losses[0],
                   loss_last=losses[-1], gradient_records=len(gradients), adapter_bytes=weights.stat().st_size)
    (directory / "training_validation.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_train(st, a):
    gpus = pick_gpus(2)
    cfgdir = ensure_configs(a.run)
    st.run("train-smoke", [TCLI, "train", cfgdir / "smoke.yaml"], gpu=gpus[0], timeout=3600)
    smoke_check = validate_training_output(ZBC / "workspace/lora" / (a.run + "-smoke"), 20)
    st.note(f"- 试训状态核验：{smoke_check}")
    st.note(f"- 试训 20 步通过（日志 logs/train-smoke.log）")
    env_gpu = ",".join(gpus)
    E["FORCE_TORCHRUN"] = "1"
    st.run("train", [TCLI, "train", cfgdir / "train.yaml"], gpu=env_gpu, timeout=72 * 3600)
    train_check = validate_training_output(ZBC / "workspace/lora" / a.run, 1)
    st.note(f"- 正式训练状态核验：{train_check}")
    st.run("export", [TCLI, "export", cfgdir / "export.yaml"], gpu=None, timeout=6 * 3600)
    adapter = ZBC / "workspace/lora" / a.run
    state = adapter / "trainer_state.json"
    if state.exists():
        logs = json.loads(state.read_text()).get("log_history", [])
        tl = [x["loss"] for x in logs if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]
        st.note(f"\n## 训练\n- 总步数 {logs[-1].get('step') if logs else '?'}；训练 loss 首/末 "
                f"{tl[0] if tl else '?'} → {tl[-1] if tl else '?'}\n- 验证 loss：{ev}")
    st.note(f"- 合并模型：`{ZBC / 'weights' / ('Qwen2.5-VL-7B-nav-' + a.run)}`；loss 曲线 `{adapter}/training_loss.png`")


def stage_closed(st, a):
    g_llm, g_sim = pick_gpus(2)
    plans = PLANS / "val_unseen.jsonl"
    out = st.dir / f"run-{a.model}"
    with model_service(st, a.run, a.model, g_llm) as cfg:
        st.run("rollout", rollout_cmd(out, a.n, cfg=cfg, plans=plans), gpu=g_sim, habitat=True, timeout=8 * 3600)
    cols = {f"{a.run}-{a.model}": summarize_run(st, out, a.model)}
    if a.compare:
        cols["对照"] = json.loads((Path(a.compare) / "metrics_val_unseen.json").read_text())
    st.note(f"\n## 闭环 {a.n} 条\n")
    st.lines += metrics_table(cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["smoke", "paired", "collect", "offline", "train", "closed"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb", type=float, default=0.15)
    p.add_argument("--run", default="r0", help="训练轮次（决定用哪套 train-configs）")
    p.add_argument("--model", default="base", choices=["base", "nav"])
    p.add_argument("--data", default="r0", help="offline：考哪份划分（decision_data/<data>-split）")
    p.add_argument("--compare", default=None, help="closed：对照的 rollout 目录")
    a = p.parse_args()
    a.tag = a.tag or (a.run if a.stage in ("train",) else f"n{a.n}")
    st = Stage(a.stage, a.tag)
    try:
        globals()[f"stage_{a.stage}"](st, a)
        st.finish(True)
    except (StageError, subprocess.CalledProcessError, FileNotFoundError, KeyError) as e:
        st.finish(False, str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        st.finish(False, "被手动中断")
        sys.exit(130)


if __name__ == "__main__":
    main()
