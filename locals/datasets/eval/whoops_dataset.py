"""WHOOPS 评测集；HuggingFace datasets 仅在实例化本类时加载，避免 import short_qa 时触发 pyarrow/datasets 依赖。"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from ..utils.box_utils import process_thought
from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN


class WhoopsDataset(Dataset):
    def __init__(self, require_cot: bool = False):
        from datasets import load_dataset

        self.meta = load_dataset("nlphuji/whoops", use_auth_token="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")[
            "test"
        ]
        self.require_cot = require_cot
        self.index2base = {}
        index = 0
        for i in range(len(self.meta)):
            item = self.meta[i]
            for j, _qa in enumerate(item["question_answering_pairs"]):
                self.index2base[index] = (i, j)
                index += 1
        self.total_num = index

    def __len__(self) -> int:
        return self.total_num

    def getlabel(self, i):
        i, j = self.index2base[i]
        return self.meta[i]["question_answering_pairs"][j][1]

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False):
        input_dict = self[i]
        all_box = [[0.0, 0.0, 1.0, 1.0]]
        new_thought, thought_boxes = process_thought(thought, mistral=mistral)
        all_box.extend(thought_boxes)
        input_dict["conversation"].append({"from": "gpt", "value": new_thought})
        input_dict["conversation"].append({"from": "human", "value": "What is your final answer?"})
        input_dict["box"] = all_box
        if thought_ids is not None:
            thought_ids = thought_ids.squeeze()
            if mistral:
                suffix = torch.tensor(
                    [
                        733,
                        16289,
                        28793,
                        1824,
                        349,
                        574,
                        1480,
                        4372,
                        28804,
                        733,
                        28748,
                        16289,
                        28793,
                    ]
                ).to(thought_ids.device)
            else:
                suffix = torch.tensor(
                    [
                        3148,
                        1001,
                        29901,
                        1724,
                        338,
                        596,
                        2186,
                        1234,
                        29973,
                        319,
                        1799,
                        9047,
                        13566,
                        29901,
                    ]
                ).to(thought_ids.device)
            del input_dict["conversation"]
            eoc_indices = [-1] + torch.where(thought_ids == eoc_id)[0].tolist() + [thought_ids.shape[0] - 1]
            input_dict["input_ids"] = []
            for i in range(len(eoc_indices) - 1):
                input_dict["input_ids"].append(thought_ids[eoc_indices[i] + 1 : eoc_indices[i + 1] + 1])
                if i < len(eoc_indices) - 2:
                    if thought_ids[eoc_indices[i + 1] + 1].item() != img_id:
                        input_dict["input_ids"].append(torch.tensor([img_id]).to(thought_ids.device))
            input_dict["input_ids"].append(suffix)
            input_dict["input_ids"] = torch.cat(input_dict["input_ids"])
        return input_dict

    def __getitem__(self, i: int) -> dict[str, Any]:
        index_base, index_ques = self.index2base[i]
        item = self.meta[index_base]
        question = item["question_answering_pairs"][index_ques][0]
        image = item["image"]
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + "\n" + question + " " + COT_ACTIVATION
        else:
            input_sent = ALL_IMG_TOKENS_STR + "\n" + question

        sources = [{"from": "human", "value": input_sent}]
        return {"input_images": [image], "conversation": sources}
