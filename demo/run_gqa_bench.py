#!/usr/bin/env python3
"""
GQA 评测：evaluate_benchmark → convert_res_to_gqa →（val）官方 eval.py。

在 VoCoT 仓库根目录执行:
  python run_gqa_bench.py
  python run_gqa_bench.py --split testdev            # 默认 5 进程/5 卡；单卡请 --nproc 1
  python run_gqa_bench.py --split val --max_samples 500 --max_new_tokens 512   # val 子集仍会跑官方 eval（指标仅对应该子集）
  python run_gqa_bench.py --split testdev --max_samples 500

默认路径来自 project.py（volcano_7b_luoruipu1_path、gqa_bench_path）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_EVAL_TOOLS = os.path.join(_REPO_ROOT, "eval", "eval_tools")
if _EVAL_TOOLS not in sys.path:
    sys.path.insert(0, _EVAL_TOOLS)

import gqa_subset_official_eval  # noqa: E402

from project import gqa_bench_path, volcano_7b_luoruipu1_path


def _subprocess_env() -> dict[str, str]:
    """子进程跑 eval/*.py 时，脚本位于 eval/ 下，默认 sys.path 不含仓库根，无法 import locals。"""
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _REPO_ROOT if not prev else f"{_REPO_ROOT}{os.pathsep}{prev}"
    return env


def _write_dataset_yaml(gqa_root: str, split: str, out_dir: str) -> tuple[str, str, str]:
    """在 out_dir 写入 GQA 数据集 YAML，返回 (yaml_path, pred_filename, questions_basename)。"""
    os.makedirs(out_dir, exist_ok=True)
    if split == "val":
        name = "GQA_bench_val"
        qfile = "val_balanced_questions.json"
        pred_name = "val_balanced_predictions.json"
    elif split == "testdev":
        name = "GQA_bench_testdev"
        qfile = "testdev_balanced_questions.json"
        pred_name = "testdev_balanced_predictions.json"
    else:
        raise ValueError("split 必须是 val 或 testdev")

    questions_path = os.path.join(gqa_root, "questions1.2", qfile)
    image_root = os.path.join(gqa_root, "images", "images")
    yaml_path = os.path.join(out_dir, f"{name}.yaml")
    body = f"""- gqa:
  target: locals.datasets.eval.short_qa.GQADataset
  params:
    path: {questions_path}
    base_path: {image_root}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(body)
    return yaml_path, pred_name, qfile


def main() -> None:
    default_store = os.path.basename(volcano_7b_luoruipu1_path.rstrip(os.sep))
    default_out = os.path.join("output", "gqa", default_store, "cot")

    p = argparse.ArgumentParser(description="GQA 完整评测（默认参数见 project.py）")
    p.add_argument("--model_path", type=str, default=volcano_7b_luoruipu1_path, help="模型目录")
    p.add_argument("--gqa_root", type=str, default=gqa_bench_path, help="GQA_Bench 数据集根目录")
    p.add_argument("--split", type=str, default="val", choices=("val", "testdev"))
    p.add_argument("--output_dir", type=str, default=default_out, help="推理与预测 JSON 输出目录")
    p.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="分布式进程数（通常等于 GPU 数）；超过本机可见 GPU 数会自动截断。1 表示单进程、不用 torch.distributed.run",
    )
    p.add_argument("--port", type=int, default=8891, help="torchrun master 端口")
    p.add_argument(
        "--skip_official_eval",
        action="store_true",
        help="val 时跳过官方 eval.py（仅推理 + 转换预测格式）",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="每轮生成的最大新 token 数；调小可明显加速，但可能截断长 CoT（默认 2048）",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="只评测前 N 条题目（0 表示全量）；用于冒烟或快速估时",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=500,
        help="传给 evaluate_benchmark：每个进程每完成多少条打印一条进度日志（0 关闭）",
    )
    args = p.parse_args()

    nproc = args.nproc
    try:
        import torch

        if torch.cuda.is_available():
            ngpu = torch.cuda.device_count()
            if nproc > ngpu:
                print(
                    f"[run_gqa_bench] 警告: --nproc={nproc} 大于本机可见 GPU 数 ({ngpu})，已改为 {ngpu}，避免 CUDA invalid device ordinal",
                    file=sys.stderr,
                )
                nproc = ngpu
        elif nproc > 1:
            print(
                "[run_gqa_bench] 警告: 未检测到 CUDA，无法多卡并行，已改为 nproc=1",
                file=sys.stderr,
            )
            nproc = 1
    except ImportError:
        if nproc > 1:
            print("[run_gqa_bench] 警告: 无法 import torch，请使用 --nproc 1", file=sys.stderr)
            nproc = 1

    yaml_path, pred_name, _qfile = _write_dataset_yaml(args.gqa_root, args.split, args.output_dir)

    flag = [
        "--model_path",
        args.model_path,
        "--eval_data",
        yaml_path,
        "--output_dir",
        args.output_dir,
        "--avoid_image_gen",
        "--temperature",
        "0",
        "--precision",
        "fp16",
        "--expand2square",
        "--use_mistral",
        "--cot",
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    if args.max_samples > 0:
        flag.extend(["--sub_sample", str(args.max_samples)])
    flag.extend(["--log_every", str(args.log_every)])

    if nproc > 1:
        # 必须用当前解释器启动分布式入口，避免 PATH 上的 torchrun 指向 /usr/local（与 conda 的 numpy/torch 不一致）
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={args.port}",
            "eval/evaluate_benchmark.py",
            *flag,
        ]
    else:
        cmd = [sys.executable, "eval/evaluate_benchmark.py", *flag]

    subprocess.check_call(cmd, cwd=_REPO_ROOT, env=_subprocess_env())

    out_base = os.path.splitext(os.path.basename(yaml_path))[0]
    src_json = os.path.join(args.output_dir, f"{out_base}.json")
    dst_json = os.path.join(args.output_dir, pred_name)

    subprocess.check_call(
        [
            sys.executable,
            "eval/eval_tools/convert_res_to_gqa.py",
            "--config",
            yaml_path,
            "--src",
            src_json,
            "--dst",
            dst_json,
        ],
        cwd=_REPO_ROOT,
        env=_subprocess_env(),
    )
    print(f"已生成官方格式预测: {dst_json}")

    if args.split == "val" and not args.skip_official_eval:
        subset_dir = os.path.join(args.output_dir, "_gqa_subset_official")
        if args.max_samples > 0:
            q_eval, c_eval, n_eff = gqa_subset_official_eval.prepare_val_balanced_subset_files(
                args.gqa_root, args.max_samples, subset_dir
            )
            eval_py = gqa_subset_official_eval.write_subset_safe_eval_script(
                args.gqa_root, subset_dir
            )
            print(
                f"运行官方 GQA eval（balanced val 子集，前 {n_eff} 题，与推理子集一致）...",
                flush=True,
            )
        else:
            q_eval = os.path.join(
                args.gqa_root, "questions1.2", "val_balanced_questions.json"
            )
            c_eval = os.path.join(args.gqa_root, "eval", "val_choices.json")
            eval_py = os.path.join(args.gqa_root, "eval", "eval.py")
            print("运行官方 GQA eval.py（balanced val 全量）...", flush=True)

        subprocess.check_call(
            [
                sys.executable,
                eval_py,
                "--tier",
                "val",
                "--scenes",
                os.path.join(args.gqa_root, "sceneGraphs", "val_sceneGraphs.json"),
                "--questions",
                q_eval,
                "--choices",
                c_eval,
                "--predictions",
                dst_json,
            ],
            cwd=_REPO_ROOT,
        )
    elif args.split == "testdev":
        print(f"testdev 无本地参考答案；{dst_json} 可用于 GQA 官方提交格式。")


if __name__ == "__main__":
    main()
