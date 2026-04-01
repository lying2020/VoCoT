from genericpath import samestat
from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from locals.datasets import SFT_DataCollator, WrappedDataset
from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from locals.datasets.preprocessor import VoCoT_InputProcessor
from omegaconf import OmegaConf
from utils.util import instantiate_from_config
from model.language_model.volcano_llama import VolCanoLlamaForCausalLM,VolCanoConfig
from model.language_model.volcano_mistral import VolCanoMistralForCausalLM, VolCanoMistralConfig
from transformers import LlamaTokenizer, AutoTokenizer
import transformers
from peft import PeftConfig, PeftModel
from argparse import ArgumentParser
import os
import torch.distributed as dist
from utils.logger import setup_logger
import json
import tqdm
import contextlib


def _resolve_local_clip_path():
    """优先环境变量 VOCOT_LOCAL_CLIP_PATH，否则使用 project.clip_vit_large_patch14_336_path（离线加载 CLIP，避免访问 huggingface.co）。"""
    p = os.environ.get("VOCOT_LOCAL_CLIP_PATH", "").strip()
    if p and os.path.isdir(p):
        return p
    try:
        from project import clip_vit_large_patch14_336_path as p2

        if p2 and os.path.isdir(p2):
            return p2
    except ImportError:
        pass
    return None


def rank0_print(args, res):
    if args.local_rank==0 or args.local_rank == -1:
        print(res)

def get_output_name(args, mid_output=True):
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}.json'.format(args.dataset_name))

def get_all_output_names(args):
    return [os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, r)) for r in range(args.n_gpus)]

class CLIPTransform:
    def __init__(self, transform, square_size=None):
        self.transform = transform
        self.square_size = square_size
        self.image_mean = transform.image_mean
    
    def __call__(self, image):
        if self.square_size is not None:
            image = image.resize((self.square_size, self.square_size))
        try:
            tmp = torch.tensor(self.transform(image)['pixel_values'][0])
        except:
            tmp = torch.tensor(self.transform(Image.new(image.mode, (32, 32), (0,0,0)))['pixel_values'][0])
        return tmp



def load_model(model_path, device='cuda:0', precision='bf16'):
    config_class = VolCanoMistralConfig
    model_class = VolCanoMistralForCausalLM
    tokenizer_class = AutoTokenizer
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        use_fast=True,
        trust_remote_code=True
    )
    
    llama_config = config_class.from_pretrained(model_path)
    local_clip = _resolve_local_clip_path()
    if local_clip and getattr(llama_config, "vision_encoder", None) is not None:
        ve = str(llama_config.vision_encoder)
        if "openai" in ve and "clip" in ve.lower():
            llama_config.vision_encoder = local_clip
    model = model_class.from_pretrained(model_path, config=llama_config)

    model.input_img_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMG_TOKEN)
    model.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
    model.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
    model.tokenizer = tokenizer
    model.sub_image_bind = False

    if precision == 'bf16':
        model.to(torch.bfloat16)
    elif precision == 'fp16':
        model.to(torch.float16)
    elif precision == 'fp32':
        pass
    else:
        raise ValueError('precision must be fp16, bf16, or fp32')
    model.eval()
    model.to(device)

    resize2square = False
    output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    # transforms.RandomHorizontalFlip(), # comment here
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    input_vis_processor = transforms.Compose(
            [
                transforms.Resize((448, 448) if resize2square else 448, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(448),
                # transforms.RandomHorizontalFlip(), comment here
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        )
    if hasattr(model.vision_encoder, 'image_processor'):
        input_vis_processor = model.vision_encoder.image_processor
        if resize2square:
            tmp_size = input_vis_processor.size['shortest_edge']
        else:
            tmp_size = None
        input_vis_processor = CLIPTransform(input_vis_processor, square_size=tmp_size)
    # tokenizer = LlamaTokenizer.from_pretrained('eval/debug/edit_gpt_emu_tokenizer')

    model.image_processor = None
    preprocessor = VoCoT_InputProcessor(tokenizer=tokenizer, input_image_processor = input_vis_processor, use_mistral=True,
                                                output_image_processor= output_vis_processor, merge_in_out_image=True, expand2square=True, inference = True)

    return model, preprocessor

def _condition_one_batch(
    model,
    preprocessor,
    item,
    max_new_tokens,
    temperature,
    avoid_image_gen=True,
    kv_tracer=None,
    kv_sample_meta=None,
):
    input_item = preprocessor(item)
    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    batch = data_collator([input_item])

    ctx = contextlib.nullcontext()
    if kv_tracer is not None:
        from utils.kv_trace import build_prefill_modality_mask

        kv_tracer.start_sample(kv_sample_meta or {})
        mask = build_prefill_modality_mask(
            batch["input_ids"].cpu(),
            model.input_img_id,
            model.n_query,
        )
        kv_tracer.set_prefill_modality(mask)
        ctx = kv_tracer

    with ctx:
        txt_res, out_imgs, txt_ids = model.condition_completion(
            batch,
            avoid_image_gen=avoid_image_gen,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    return txt_res, out_imgs, txt_ids


def infer(
    model,
    preprocessor,
    image,
    query,
    cot=True,
    max_new_tokens=1024,
    temperature=0.0,
    avoid_image_gen=True,
    kv_tracer=None,
    kv_sample_meta=None,
    return_full=False,
):
    if cot:
        query = (
            ALL_IMG_TOKENS_STR
            + DEFAULT_GRD_TOKEN
            + "\n"
            + query
            + COT_ACTIVATION
        )
    else:
        query = ALL_IMG_TOKENS_STR + "\n" + query
    conv = [{"from": "human", "value": query}]
    item = {"input_images": [image], "conversation": conv}
    pack = _condition_one_batch(
        model,
        preprocessor,
        item,
        max_new_tokens,
        temperature,
        avoid_image_gen=avoid_image_gen,
        kv_tracer=kv_tracer,
        kv_sample_meta=kv_sample_meta,
    )
    if return_full or kv_tracer is not None:
        return pack
    return pack[0]


def infer_multiturn(
    model,
    preprocessor,
    image,
    turns,
    cot_last=True,
    max_new_tokens=1024,
    temperature=0.0,
):
    """
    多轮对话式推理：同一张图，带历史 user/assistant 轮次，只生成最后一轮 assistant。

    turns: list[tuple[str, str]]，每项为 (role, text)，role 为 'user' 或 'assistant'。
    须以 user 开始、user 结束，且严格交替（user, assistant, user, ...）。

    首轮 user 自动加 <ImageHere> + <grounding>；仅在最后一轮 user 上附加 VoCoT COT 提示（cot_last=True）。
    """
    if not turns or turns[0][0] != 'user' or turns[-1][0] != 'user':
        raise ValueError("turns 须非空、以 user 开始并以 user 结束")
    for i, (role, _) in enumerate(turns):
        expect = 'user' if i % 2 == 0 else 'assistant'
        if role != expect:
            raise ValueError(f"turns[{i}] 应为 {expect}，得到 {role}")

    conv = []
    n = len(turns)
    for i, (role, text) in enumerate(turns):
        if role == 'user':
            if i == 0:
                text = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + text
            if i == n - 1 and cot_last:
                text = text + COT_ACTIVATION
            conv.append({'from': 'human', 'value': text})
        else:
            conv.append({'from': 'gpt', 'value': text})

    item = {'input_images': [image], 'conversation': conv}
    return _condition_one_batch(
        model, preprocessor, item, max_new_tokens, temperature, avoid_image_gen=True,
    )[0]


def infer_multihop_document(
    model,
    preprocessor,
    image,
    query,
    cot=True,
    hop_instruction=None,
    max_new_tokens=2048,
    temperature=0.0,
):
    """
    单次生成多 HOP 的「文档式」输出：在问题后附加 MULTIHOP_DOC_INSTRUCTION（可自定义），再走 infer。
    适合需要分节、多跳推理链但不要求真实多轮对话的场景。
    """
    extra = hop_instruction if hop_instruction is not None else MULTIHOP_DOC_INSTRUCTION
    full_query = query.rstrip() + '\n\n' + extra
    return infer(model, preprocessor, image, full_query, cot=cot, max_new_tokens=max_new_tokens, temperature=temperature)
            

if __name__=='__main__':
    from PIL import Image
    tmp_image = Image.open('eval/debug/tmp.jpg')
    model_path = '/mnt/bn/yangmin-priv/luoruipu/checkpoints/LLaVA-clip336px-obj-represent-Mistral-1e-5-3072-instruct_llava+shikraCoT75per+GPTQTA+lvis-cot/'
    model, preprocessor = load_model(model_path,precision='fp16')
    res1 = infer(model, preprocessor, tmp_image, 'Is there a event "the cat is below the bed" in this image?', cot=True)
    res = infer(model, preprocessor, tmp_image, 'Why is the cat on the bed?', cot=True)
    res_no_cot = infer(model, preprocessor, tmp_image, 'Describe the image.', cot=True)
    print(res1)
    print(res)
    print(res_no_cot)