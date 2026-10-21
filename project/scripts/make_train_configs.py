"""为某一轮训练生成全部本机路径的配置（在训练环境 havln-train 里运行，需要 pyyaml）。

  python scripts/make_train_configs.py --run r0 --root /mnt/data/zbc \
      --dataset /mnt/data/zbc/workspace/decision_data/r0-split

输出目录 <root>/workspace/train-configs/<run>/：
  smoke.yaml      20 个优化步、最多 64 条样本：只检查数据能读、loss 正常、显存装得下、checkpoint 能存
  train.yaml      正式 QLoRA 训练（参数来自 configs/train_decision_lora.yaml，只替换路径）
  export.yaml     在 CPU 上把 LoRA 合并回原版 bf16 权重
  api-base.yaml   用 LLaMA-Factory 的 HF 后端起原版模型的 OpenAI 兼容服务（4-bit）
  api-nav.yaml    同上，起本轮合并后的模型
  agent-base.json / agent-nav.json  智能体连接上面两个服务的配置（模型名、缓存目录各自独立）
目录已存在就拒绝覆盖：每一轮的配置原样留档，结果才能复现。
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="轮次名，如 r0、r1")
    p.add_argument("--root", default="/mnt/data/zbc")
    p.add_argument("--dataset", required=True, help="gen_decision_data.py --merge 的输出目录")
    p.add_argument("--base-model", default=None, help="默认 <root>/weights/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--num-gpus", type=int, default=2, help="正式训练用几张卡；梯度累积按 16 / 卡数 设置，保持等效批量 16")
    args = p.parse_args()

    root = Path(args.root).resolve()
    run = args.run
    dest = root / "workspace/train-configs" / run
    if dest.exists():
        raise SystemExit(f"{dest} 已存在；换一个 --run 名，或确认后手动删除")
    base = Path(args.base_model or root / "weights/Qwen2.5-VL-7B-Instruct").resolve()
    dataset = Path(args.dataset).resolve()
    adapter = root / "workspace/lora" / run
    merged = root / "weights" / f"Qwen2.5-VL-7B-nav-{run}"
    if not (base / "config.json").is_file():
        raise SystemExit(f"缺基座模型 {base}/config.json")
    if not (dataset / "dataset_info.json").is_file():
        print(f"提示：{dataset} 还没有划分好的数据。服务和智能体配置可以先用，训练前先跑 gen_decision_data.py --merge")

    train = yaml.safe_load((ROOT / "configs/train_decision_lora.yaml").read_text())
    train.update(model_name_or_path=str(base), dataset_dir=str(dataset), output_dir=str(adapter),
                 overwrite_output_dir=False, report_to="none", seed=0, data_seed=0,
                 quantization_method="bitsandbytes", double_quantization=True, quantization_type="nf4",
                 gradient_accumulation_steps=max(1, 16 // args.num_gpus), ddp_find_unused_parameters=False)
    smoke = dict(train, output_dir=str(adapter) + "-smoke", max_steps=20, logging_steps=1,
                 save_steps=20, eval_steps=20, gradient_accumulation_steps=1, max_samples=64)
    export = yaml.safe_load((ROOT / "configs/export_merged.yaml").read_text())
    export.update(model_name_or_path=str(base), adapter_name_or_path=str(adapter), export_dir=str(merged))

    dest.mkdir(parents=True)
    for name, value in (("train", train), ("smoke", smoke), ("export", export)):
        (dest / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    for name, weight in (("base", base), ("nav", merged)):
        api = dict(model_name_or_path=str(weight), template="qwen2_vl", infer_backend="huggingface",
                   trust_remote_code=True, quantization_bit=4, quantization_method="bitsandbytes",
                   image_max_pixels=112896, do_sample=False, max_new_tokens=1024)
        (dest / f"api-{name}.yaml").write_text(yaml.safe_dump(api, sort_keys=False))
        agent = {"slow": {"base_url": f"http://127.0.0.1:{args.port}/v1", "model": f"zbc-{run}-{name}",
                          "cache_dir": str(root / "cache" / f"llm-{run}-{name}"),
                          "use_env_proxy": False, "timeout_s": 180.0}}
        (dest / f"agent-{name}.json").write_text(json.dumps(agent, indent=2))
    print(f"配置写到 {dest}")
    print(f"  服务模型名：zbc-{run}-base / zbc-{run}-nav（起 API 时 API_MODEL_NAME 要与之一致）")


if __name__ == "__main__":
    main()
