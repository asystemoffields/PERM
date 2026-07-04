"""qwen3 spacemap: G3 logits-equality on a TINY random Qwen3 (identity => exact 0, random
perms => rel < 1e-4), consume-mode apply_perms equality, and input_perm shapes."""
import copy

import numpy as np
import pytest

from coldpress.perm.spacemaps import qwen3


def make_random_perms(dims, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "res": rng.permutation(dims["d_model"]),
        "ffn": [rng.permutation(dims["d_ffn"]) for _ in range(dims["n_layers"])],
        "vo": [[rng.permutation(dims["head_dim"]) for _ in range(dims["n_kv"])]
               for _ in range(dims["n_layers"])],
    }


def test_g3_identity_exact_zero(tiny_qwen3, tiny_ids):
    import torch
    model, cfg = tiny_qwen3
    dims = qwen3.dims_from_config(cfg)
    m = copy.deepcopy(model)
    d, rel = qwen3.g3_check(m, qwen3.identity_perms(dims), dims, tiny_ids)
    assert d == 0.0, f"identity perms drifted logits by {d}"


def test_g3_random_perms_preserves_logits(tiny_qwen3, tiny_ids):
    import torch
    model, cfg = tiny_qwen3
    dims = qwen3.dims_from_config(cfg)
    m = copy.deepcopy(model)
    perms = make_random_perms(dims, seed=3)
    d, rel = qwen3.g3_check(m, perms, dims, tiny_ids)
    assert rel < 1e-4, f"G3 FAIL: rel={rel:.3e} max|dlogit|={d:.3e}"


def test_consume_mode_equality(tiny_qwen3):
    import torch
    model, cfg = tiny_qwen3
    dims = qwen3.dims_from_config(cfg)
    perms = make_random_perms(dims, seed=5)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out_a = qwen3.apply_perms({k: v.clone() for k, v in sd.items()}, perms, dims, consume=False)
    out_b = qwen3.apply_perms({k: v.clone() for k, v in sd.items()}, perms, dims, consume=True)
    assert set(out_a) == set(out_b)
    for k in out_a:
        assert torch.equal(out_a[k], out_b[k]), f"consume-mode differs at {k}"


def test_input_perm_shapes(tiny_qwen3):
    model, cfg = tiny_qwen3
    dims = qwen3.dims_from_config(cfg)
    perms = make_random_perms(dims, seed=7)
    assert qwen3.input_perm("token_embd.weight", perms, dims).shape == (dims["d_model"],)
    assert qwen3.input_perm("blk.0.attn_q.weight", perms, dims).shape == (dims["d_model"],)
    assert qwen3.input_perm("blk.1.ffn_down.weight", perms, dims).shape == (dims["d_ffn"],)
    aout = qwen3.input_perm("blk.0.attn_output.weight", perms, dims)
    assert aout.shape == (dims["n_heads"] * dims["head_dim"],)
    # it is a valid permutation
    assert np.array_equal(np.sort(aout), np.arange(dims["n_heads"] * dims["head_dim"]))
    assert qwen3.input_perm("output_norm.weight", perms, dims) is None
