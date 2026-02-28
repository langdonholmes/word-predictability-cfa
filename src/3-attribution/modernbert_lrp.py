"""AttnLRP patches for ModernBERT via LXT's efficient framework.

Applies the same LRP rules as LXT's built-in BERT support:
  - divide_gradient(_, 2) on attention matmuls (uniform rule)
  - identity_rule_implicit on nonlinear activations (identity rule)
  - stop_gradient on LayerNorm statistics (identity rule via global patch)
  - dropout disabled

Usage:
    import transformers.models.modernbert.modeling_modernbert as modeling_mb
    from modernbert_lrp import monkey_patch_modernbert
    monkey_patch_modernbert(modeling_mb, verbose=True)
    model = modeling_mb.ModernBertForMaskedLM.from_pretrained(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from lxt.efficient.rules import divide_gradient, identity_rule_implicit
from lxt.efficient.patches import patch_method, layer_norm_forward, dropout_forward


def patched_eager_attention_forward(
    module,
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
    attn_weights = divide_gradient(attn_weights, 2)  # LRP: uniform rule on Q×K

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    # LRP: dropout disabled
    attn_output = torch.matmul(attn_weights, value)
    attn_output = divide_gradient(attn_output, 2)  # LRP: uniform rule on Attn×V

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def patched_sdpa_attention_forward(
    module,
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    **_kwargs,
) -> tuple[torch.Tensor]:
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    query = divide_gradient(query, 2)  # LRP: uniform rule
    key = divide_gradient(key, 2)      # LRP: uniform rule

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0,  # LRP: dropout disabled
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = divide_gradient(attn_output, 2)  # LRP: uniform rule on output

    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


def patched_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
    input = identity_rule_implicit(self.act, input)  # LRP: identity rule on activation
    weighted = input * gate
    weighted = divide_gradient(weighted, 2)  # LRP: uniform rule on gate × input
    return self.Wo(self.drop(weighted))


def patched_prediction_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = identity_rule_implicit(self.act, hidden_states)  # LRP: identity rule
    return self.norm(hidden_states)


def monkey_patch_modernbert(module, verbose=False):
    """Apply AttnLRP patches to the ModernBERT modeling module.

    Must be called BEFORE loading the model with from_pretrained().

    Parameters
    ----------
    module : the transformers.models.modernbert.modeling_modernbert module
    verbose : bool, print patched components
    """
    # Global patches: LayerNorm and Dropout
    patch_method(layer_norm_forward, torch.nn.LayerNorm)
    if verbose:
        print("Patched LayerNorm")
    patch_method(dropout_forward, torch.nn.Dropout)
    if verbose:
        print("Patched Dropout")

    # Patch attention function dispatch table
    module.MODERNBERT_ATTENTION_FUNCTION["eager"] = patched_eager_attention_forward
    module.MODERNBERT_ATTENTION_FUNCTION["sdpa"] = patched_sdpa_attention_forward
    if verbose:
        print("Patched ModernBERT attention (eager + sdpa)")

    # Patch MLP
    module.ModernBertMLP.forward = patched_mlp_forward
    if verbose:
        print("Patched ModernBertMLP")

    # Patch prediction head
    module.ModernBertPredictionHead.forward = patched_prediction_head_forward
    if verbose:
        print("Patched ModernBertPredictionHead")
