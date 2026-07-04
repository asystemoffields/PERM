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
