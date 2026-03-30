# model_utils/dare/utils.py
import torch


def build_modality_mask(
    input_ids: torch.LongTensor,
    image_token_ids,
) -> torch.LongTensor:
    if not isinstance(image_token_ids, torch.Tensor):
        image_token_ids = torch.tensor(
            image_token_ids, device=input_ids.device, dtype=input_ids.dtype
        )
    else:
        image_token_ids = image_token_ids.to(device=input_ids.device, dtype=input_ids.dtype)

    is_image = (input_ids.unsqueeze(-1) == image_token_ids.view(1, 1, -1)).any(dim=-1)

    # 0 = text, 1 = vision
    modality_mask = is_image.to(dtype=torch.long)    # or bool if you prefer

    return modality_mask
