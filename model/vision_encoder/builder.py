import os
from .clip_encoder import CLIPVisionTower
from .eva_vit import create_eva_vit_g
from .eva_vit_emu import create_eva_vit_emu

def build_vision_encoder(vision_encoder_cfg, **kwargs):
    vision_tower = getattr(vision_encoder_cfg, 'vision_encoder', None)
    is_absolute_path_exists = os.path.exists(vision_tower)

    # 本地已下载的 HuggingFace 格式 CLIP 目录（含 config.json 等），无需访问 huggingface.co
    _local_clip = (
        is_absolute_path_exists
        and os.path.isdir(vision_tower)
        and ("clip" in os.path.basename(vision_tower).lower())
    )
    if vision_tower.startswith("openai") or (is_absolute_path_exists and (vision_tower.startswith("OFA-Sys") or vision_tower.startswith("laion"))) or _local_clip:
        return CLIPVisionTower(vision_tower, args=vision_encoder_cfg, **kwargs)
    elif is_absolute_path_exists and 'eva' in vision_tower:
        return create_eva_vit_g(vision_encoder_cfg.vision_encoder, **kwargs)
    elif vision_tower == 'eva_vit_emu':
        # using the emu-pre-trained vit
        vision_encoder_path = getattr(vision_encoder_cfg, 'vision_encoder_path', None)
        load_encoder_ckpt = not getattr(vision_encoder_cfg, 'skip_load_vision_encoder', False)
        assert vision_encoder_path is not None, 'please specify the model path for emu-pre-trained vision encoder'
        return create_eva_vit_emu(vision_encoder_path, load_ckpt=load_encoder_ckpt)
    raise ValueError(f'Unknown vision tower: {vision_tower}')
