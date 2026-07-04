"""gemma4 spacemap strip_vision: the multimodal guard raises without the flag, and
strips vision-tower/projector tensors (then completes the text-stack permutation) with it.

Synthetic, tiny, torch-only (no model download). head_dim is the map's hardcoded _HD=256."""
import numpy as np
import pytest
import torch

from coldpress.perm.spacemaps import gemma4

_HD = gemma4._HD  # 256


def _tiny_dims():
    return {
        "d_model": 8, "d_ffn": 16, "n_layers": 1,
        "n_kv": 2, "n_heads": 4, "head_dim": _HD,
        "sliding_layers": [0], "global_layers": [],
    }


def _identity_perms():
    return {
        "res": np.arange(8),
        "ffn": [np.arange(16)],
        "vo": {0: [np.arange(_HD) for _ in range(2)]},
    }


def _tiny_gemma4_sd(with_vision=True):
    """A structurally-complete 1-(sliding-)layer gemma4 text stack (+ optional vision keys)."""
    def w(*shape):
        return torch.randn(*shape)
    sd = {
        "model.embed_tokens.weight": w(5, 8),
        "model.norm.weight": w(8),
        "model.layers.0.input_layernorm.weight": w(8),
        "model.layers.0.post_attention_layernorm.weight": w(8),
        "model.layers.0.pre_feedforward_layernorm.weight": w(8),
        "model.layers.0.post_feedforward_layernorm.weight": w(8),
        "model.layers.0.mlp.gate_proj.weight": w(16, 8),
        "model.layers.0.mlp.up_proj.weight": w(16, 8),
        "model.layers.0.mlp.down_proj.weight": w(8, 16),
        "model.layers.0.self_attn.q_proj.weight": w(4 * _HD, 8),
        "model.layers.0.self_attn.k_proj.weight": w(2 * _HD, 8),
        "model.layers.0.self_attn.v_proj.weight": w(2 * _HD, 8),
        "model.layers.0.self_attn.o_proj.weight": w(8, 4 * _HD),
    }
    if with_vision:
        sd["model.vision_tower.encoder.layers.0.weight"] = w(4, 4)
        sd["model.multi_modal_projector.linear.weight"] = w(8, 4)
    return sd


def test_guard_raises_without_strip_vision():
    sd = _tiny_gemma4_sd(with_vision=True)
    with pytest.raises(NotImplementedError, match="vision"):
        gemma4.apply_perms(sd, _identity_perms(), _tiny_dims(), consume=True,
                           acknowledge_unreviewed=True, strip_vision=False)


def test_strip_vision_drops_tensors_and_completes():
    sd = _tiny_gemma4_sd(with_vision=True)
    ref = {k: v.clone() for k, v in sd.items()}
    out = gemma4.apply_perms(sd, _identity_perms(), _tiny_dims(), consume=True,
                             acknowledge_unreviewed=True, strip_vision=True)
    # vision tensors are gone
    assert not [k for k in out if "vision" in k or "multi_modal_projector" in k]
    # text stack survives; identity perms => values unchanged
    for k in ("model.embed_tokens.weight", "model.norm.weight",
              "model.layers.0.self_attn.v_proj.weight",
              "model.layers.0.self_attn.o_proj.weight"):
        assert k in out
        assert torch.equal(out[k], ref[k]), f"identity perm changed {k}"


def test_no_vision_tensors_is_a_noop_passthrough():
    """No vision keys => no strip needed, no raise, regardless of the flag."""
    sd = _tiny_gemma4_sd(with_vision=False)
    out = gemma4.apply_perms(sd, _identity_perms(), _tiny_dims(), consume=True,
                             acknowledge_unreviewed=True, strip_vision=False)
    assert "model.embed_tokens.weight" in out
