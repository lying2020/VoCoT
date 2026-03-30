from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from .controller import DAREController



class DAREWrappedBlock(nn.Module):
    def __init__(self, base_block: nn.Module, dare_controller: DAREController, layer_idx: int):
        super().__init__()
        self.block = base_block
        self.controller = dare_controller
        self.layer_idx = layer_idx

        hidden_size = None
        if hasattr(base_block, "self_attn"):
            attn = base_block.self_attn
            hidden_size = getattr(attn, "embed_dim", None) \
                       or getattr(attn, "hidden_size", None)

        hidden_size = hidden_size or getattr(base_block, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("[DARE] Could not infer hidden_size for importance scoring.")

        self.importance_proj = nn.Linear(hidden_size, 1)


    def _compute_importance(self, hidden_states):
        scores = self.importance_proj(hidden_states)      # [B,T,1]
        return scores.squeeze(-1)                         # [B,T]


    def forward(
        self,
        hidden_states: torch.Tensor,          
        modality_mask: torch.Tensor,          
        hop_idx: int,                         
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        training: Optional[bool] = None,
    ):
        if training is None:
            training = self.training

        # ----------------------
        # 1. Compute importance
        # ----------------------
        normed = self.block.ln1(hidden_states)
        importance = self._compute_importance(normed)

        text_mask = (modality_mask == 0)
        vis_mask  = (modality_mask == 1)

        # ---- route text ----
        if text_mask.any():
            keep_text, loss_text = self.controller.route(
                importance=importance,
                modality_mask=text_mask,
                is_visual=False,
                layer_idx=self.layer_idx,
                training=training,
            )
        else:
            keep_text = torch.zeros_like(text_mask)
            loss_text = {k: importance.new_zeros(()) for k in [
                "dare_ratio_loss","dare_soft_loss","dare_hard_loss"
            ]}

        # ---- route vision ----
        if vis_mask.any():
            keep_vis, loss_vis = self.controller.route(
                importance=importance,
                modality_mask=vis_mask,
                is_visual=True,
                layer_idx=self.layer_idx,
                training=training,
            )
        else:
            keep_vis = torch.zeros_like(vis_mask)
            loss_vis = {k: importance.new_zeros(()) for k in [
                "dare_ratio_loss","dare_soft_loss","dare_hard_loss"
            ]}

        keep_mask = keep_text | keep_vis
        exec_mask = keep_mask

        attn_out, _, new_past_kv = self.block.self_attn(
            normed,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            keep_mask=keep_mask,
        )

        # Residual
        hidden_states = hidden_states + attn_out


        mlp_out = self.block.mlp(self.block.ln2(hidden_states))
        hidden_states = hidden_states + mlp_out

        hidden_states = hidden_states * exec_mask.unsqueeze(-1)

        aux_losses = {
            "dare_ratio_loss": loss_text["dare_ratio_loss"] + loss_vis["dare_ratio_loss"],
            "dare_soft_loss":  loss_vis["dare_soft_loss"],
            "dare_hard_loss":  loss_text["dare_hard_loss"] + loss_vis["dare_hard_loss"],
        }

        return hidden_states, new_past_kv, exec_mask, aux_losses
