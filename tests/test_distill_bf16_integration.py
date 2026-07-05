"""Distillation end-to-end at float32 AND bfloat16 (SCALE blocker #1 integration): both the
NORM and E3B skeletons run on a tiny real Qwen3, the frozen backbone loads in the requested
dtype, the trainable scale/norm params stay float32, and validation KL is finite and never
increases (best <= before). This is the integration check behind the reconstruct/grad unit
test -- it exercises the functional_call skeleton, the bf16 casts, and the GGUF write-back."""
import os

import numpy as np
import pytest
import gguf
from gguf import GGUFWriter, get_tensor_name_map

from coldpress import kquant as kq
from coldpress import distill
from coldpress.hessians import arch_to_enum


def _build_quant_gguf(path, model, arch="qwen3", n_layers=2, tt="Q4_K"):
    """Quantize the tiny model's 2D weights (Q4_K) + keep 1D norms F32, under qwen3 GGUF
    names, so distill.dequant_state_dict can rebuild an HF state dict from it."""
    nm = get_tensor_name_map(arch_to_enum(arch), n_layers)
    w = GGUFWriter(path, arch)
    w.add_uint32(f"{arch}.block_count", n_layers)
    written = set()

    def put(gname, tensor):
        arr = tensor.detach().float().numpy()
        if arr.ndim == 2 and arr.shape[1] % 256 == 0:
            raw = kq.QUANTIZE[tt](np.ascontiguousarray(arr, np.float32), None)
            w.add_tensor(gname, np.ascontiguousarray(raw, np.uint8),
                         raw_shape=raw.shape, raw_dtype=getattr(gguf.GGMLQuantizationType, tt))
        else:
            w.add_tensor(gname, arr.astype(np.float32))
        written.add(gname)

    for hf, p in model.named_parameters():
        stem = hf[:-len(".weight")] if hf.endswith(".weight") else hf
        g = nm.get_name(stem)
        if g is None:
            continue
        put(g + ".weight", p)
    if "output.weight" not in written:                      # tied model: add an output tensor
        put("output.weight", model.get_input_embeddings().weight)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def _teacher_chunks(model, tdir, n=3, ctx=16, topk=8):
    import torch
    os.makedirs(tdir, exist_ok=True)
    rng = np.random.default_rng(0)
    for c in range(n):
        ids = torch.from_numpy(rng.integers(0, model.config.vocab_size, ctx)).long()
        with torch.inference_mode():
            lp = torch.log_softmax(model(ids.unsqueeze(0)).logits[0].float(), -1)
            top_lp, top_i = lp.topk(topk, -1)
            tail = torch.log1p(-top_lp.exp().sum(-1).clamp(max=1 - 1e-7))
        np.savez(os.path.join(tdir, f"chunk{c:04d}.npz"),
                 ids=ids.numpy(), top_i=top_i.numpy().astype(np.int32),
                 top_lp=top_lp.numpy().astype(np.float16), tail=tail.numpy().astype(np.float16))


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("which", ["norm", "e3b"])
def test_distill_runs_and_holds_kl(tiny_qwen3, tmp_path, dtype_name, which):
    import torch
    model, cfg = tiny_qwen3
    dtype = getattr(torch, dtype_name)
    model_dir = str(tmp_path / "model")
    model.save_pretrained(model_dir)
    gguf_path = str(tmp_path / "q.gguf")
    _build_quant_gguf(gguf_path, model, arch="qwen3", n_layers=cfg.num_hidden_layers)
    tdir = str(tmp_path / "teacher")
    _teacher_chunks(model, tdir)
    out = str(tmp_path / f"{which}-{dtype_name}.gguf")

    fn = distill.distill_norm if which == "norm" else distill.distill_e3b
    kw = dict(steps=6, log=lambda *_: None, dtype=dtype)
    if which == "e3b":
        kw["device"] = "cpu"
    res = fn(model_dir, "qwen3", cfg.num_hidden_layers, gguf_path, tdir, out, **kw)

    assert os.path.exists(out)
    assert np.isfinite(res["val_before"]) and np.isfinite(res["val_after"])
    assert res["val_after"] <= res["val_before"] + 1e-6, res


# ---------------------------------------------------------------- streamed student load

@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_streamed_student_equals_old_path(tiny_qwen3, tmp_path, dtype_name):
    """SCALE fix backward-compat: the per-tensor streaming loader must build the SAME student
    as the old full-f32-dequant + load_state_dict path -- logits equal within skeleton-dtype
    eps on the tiny model (both paths round f32 -> dtype identically, so expect ~exact)."""
    import torch
    model, cfg = tiny_qwen3
    dtype = getattr(torch, dtype_name)
    model_dir = str(tmp_path / "model")
    model.save_pretrained(model_dir)
    gguf_path = str(tmp_path / "q.gguf")
    _build_quant_gguf(gguf_path, model, arch="qwen3", n_layers=cfg.num_hidden_layers)

    # OLD path: materialize the full f32 dequant sd, cast, load_state_dict(strict=False)
    m_old = distill._make_untied_student(model_dir, dtype=dtype)
    gguf2hf, _, norm_map = distill.build_name_maps(m_old, "qwen3", cfg.num_hidden_layers)
    sd = {k: v.to(dtype) for k, v in distill.dequant_state_dict(gguf_path, gguf2hf).items()}
    missing, unexpected = m_old.load_state_dict(sd, strict=False)
    assert not unexpected

    # NEW path: stream one tensor at a time into the skeleton storages
    m_new = distill._make_untied_student(model_dir, dtype=dtype)
    f32_norms = distill.stream_student_from_gguf(m_new, gguf_path, gguf2hf,
                                                 keep_f32=set(norm_map),
                                                 log=lambda *_: None)
    assert set(f32_norms) == set(norm_map)
    for hf in norm_map:      # captured pre-cast: exact f32 (norms are F32 in the GGUF)
        assert f32_norms[hf].dtype == torch.float32

    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        la = m_old(ids).logits.float()
        lb = m_new(ids).logits.float()
    eps = torch.finfo(dtype).eps
    tol = eps * float(la.abs().max()) + 1e-8
    d = float((la - lb).abs().max())
    assert d <= tol, f"streamed vs old-path student logits differ: {d:.3e} > {tol:.3e}"


def test_stream_strict_accounting(tiny_qwen3, tmp_path):
    """The streaming loader enforces load_state_dict-strict semantics itself: a GGUF tensor
    with no mapping / no parameter home is a hard error, and a parameter left uncovered is a
    hard error."""
    import torch
    import pytest as _pytest
    model, cfg = tiny_qwen3
    model_dir = str(tmp_path / "model")
    model.save_pretrained(model_dir)
    gguf_path = str(tmp_path / "q.gguf")
    _build_quant_gguf(gguf_path, model, arch="qwen3", n_layers=cfg.num_hidden_layers)
    m = distill._make_untied_student(model_dir, dtype=torch.float32)
    gguf2hf, _, _ = distill.build_name_maps(m, "qwen3", cfg.num_hidden_layers)

    bad = dict(gguf2hf)
    del bad["blk.0.attn_q.weight"]                     # unmapped GGUF tensor
    with _pytest.raises(KeyError, match="name-map gap"):
        distill.stream_student_from_gguf(m, gguf_path, bad, log=lambda *_: None)

    bad = dict(gguf2hf)
    bad["blk.0.attn_q.weight"] = "not.a.parameter"     # mapped to a nonexistent param
    with _pytest.raises(KeyError, match="strict accounting"):
        distill.stream_student_from_gguf(m, gguf_path, bad, log=lambda *_: None)

    bad = dict(gguf2hf)
    bad["output.weight"] = gguf2hf["token_embd.weight"]  # lm_head never covered
    with _pytest.raises(AssertionError, match="strict accounting"):
        distill.stream_student_from_gguf(m, gguf_path, bad, log=lambda *_: None)
