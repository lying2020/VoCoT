import os
import json
import argparse
from omegaconf import OmegaConf
from utils.util import instantiate_from_config

parser = argparse.ArgumentParser(description="将 evaluate_benchmark 输出的 JSON 转为 GQA 官方 eval 所需的 predictions 格式。")
parser.add_argument("--src", type=str, required=True, help="evaluate_benchmark 产出的 *.json（列表，每项含 item_id、prediction）")
parser.add_argument("--dst", type=str, required=True, help="输出路径，写入 [{questionId, prediction}, ...]")
parser.add_argument(
    "--config",
    type=str,
    default="config/datasets/eval/GQA.yaml",
    help="与推理时相同的 GQA 数据集 YAML（用于按 item_id 下标对齐 question id）",
)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)
ds = instantiate_from_config(cfg[0])

all_answers = []
res = json.load(open(args.src))
for line in res:
    index = int(line['item_id'].split('_')[-1])
    question_id = ds.keys[index]
    text = line['prediction'].replace('</s>', '').rstrip('.').strip().lower()
    all_answers.append({"questionId": question_id, "prediction": text})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)