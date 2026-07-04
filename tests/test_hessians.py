"""Generic Hessian collector: maps every nn.Linear to its GGUF tensor name via
TensorNameMap, keys H by that name, cross-checks H.shape[0] == ne0 (hard-errors on
mismatch), and load_hessian reads both the new per-tensor and old family-keyed layouts."""
import numpy as np
import pytest

from coldpress import hessians as hess


def _gguf_ne0(model):
    """{gguf_name(.weight): in_features} for every mapped Linear in the tiny model."""
    import torch.nn as nn
    nm = hess.build_name_map(model, "qwen3", 2)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in nm:
            out[nm[name] + ".weight"] = mod.in_features
    return out


def test_collect_keys_and_shapes(tiny_qwen3):
    import torch
    model, cfg = tiny_qwen3
    ne0 = _gguf_ne0(model)
    batches = [torch.randint(0, 512, (32,)) for _ in range(2)]
    H, unmapped = hess.collect_hessians(model, batches, ne0, "qwen3", 2, log=lambda *_: None)
    # every hooked matmul is keyed by its GGUF tensor name and shape-matches ne0
    assert "blk.0.attn_q.weight" in H
    assert "blk.1.ffn_down.weight" in H
    assert "output.weight" in H  # lm_head maps to output
    for gname, h in H.items():
        assert h["H"].shape == (ne0[gname], ne0[gname])
        assert h["n"] > 0
        assert np.allclose(h["H"], h["H"].T, atol=1e-3)  # X^T X is symmetric


def test_shape_mismatch_is_hard_error(tiny_qwen3):
    import torch
    model, cfg = tiny_qwen3
    ne0 = _gguf_ne0(model)
    ne0["blk.0.attn_q.weight"] = 999  # deliberately wrong -> must refuse to mis-key
    batches = [torch.randint(0, 512, (16,))]
    with pytest.raises(ValueError, match="ne\\[0\\]|mis-key|MISMATCH"):
        hess.collect_hessians(model, batches, ne0, "qwen3", 2, log=lambda *_: None)


def test_save_and_load_roundtrip(tiny_qwen3, tmp_path):
    import torch
    model, cfg = tiny_qwen3
    ne0 = _gguf_ne0(model)
    batches = [torch.randint(0, 512, (16,))]
    H, _ = hess.collect_hessians(model, batches, ne0, "qwen3", 2, log=lambda *_: None)
    hess.save_hessians(H, str(tmp_path))
    Hload, n = hess.load_hessian(str(tmp_path), "blk.0.attn_q.weight")
    assert Hload.shape == (ne0["blk.0.attn_q.weight"],) * 2
    assert hess.has_hessian(str(tmp_path), "blk.0.attn_q.weight")


def test_backward_compat_family_layout(tmp_path):
    """load_hessian falls back to the OLD family-keyed layout (blk.L.qkv_in.npz)."""
    d = 16
    Hfam = np.eye(d, dtype=np.float32) * 3
    np.savez_compressed(str(tmp_path / "blk_0_qkv_in.npz"), H=Hfam, n=100)
    np.savez_compressed(str(tmp_path / "lmhead_in.npz"), H=Hfam, n=50)
    # per-tensor file absent -> resolves via family fallback
    H, n = hess.load_hessian(str(tmp_path), "blk.0.attn_k.weight")
    assert n == 100 and np.array_equal(H.astype(np.float32), Hfam)
    H2, n2 = hess.load_hessian(str(tmp_path), "output.weight")
    assert n2 == 50
    with pytest.raises(FileNotFoundError):
        hess.load_hessian(str(tmp_path), "blk.5.ffn_down.weight")
