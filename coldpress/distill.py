#!/usr/bin/env python3
"""Distillation of the standard container's free continuous parameters against the FP
teacher (KL on calibration text). The output GGUF stays 100% standard.

  * NORM (fast): tune only the F32 norm-gain tensors (~65K params at 0.6B). GGUF
    instantiation of Norm Tweaking (Li et al., AAAI 2024).
  * E3B (full): tune every quantized tensor's per-superblock fp16 d/dmin (integer codes
    frozen; dequant is linear in d/dmin via scale_tune) PLUS all F32 norm gains.

Untied student (contract hard rule 2): the GGUF stores token_embd and output separately,
often at different quant types, so the student is built with tie_word_embeddings=False --
a tied student would clobber one role with the other on load.

Name maps (GGUF<->HF, which params are norm gains) are built GENERICALLY from the loaded
model via gguf's TensorNameMap -- no hardcoded per-arch tables.

STUDENT CONSTRUCTION IS STREAMED (scale fix): the GGUF is dequantized ONE tensor at a time
straight into the skeleton's parameter storages (stream_student_from_gguf), so peak extra
RAM = one tensor (~2 GB at 12B) instead of the full f32 sd (~4 B/param = ~48 GB at 12B).
Strict accounting replaces load_state_dict(strict): every GGUF tensor must land in exactly
one shape-matched parameter, and every parameter must be covered (rope/rotary tables
excepted). Both NORM and E3B then train through PARTIAL functional_call overrides.
"""
import glob as _glob
import json
import os

import numpy as np

from . import kquant as kq
from . import scale_tune
from .ggufio import reemit
from .hessians import arch_to_enum


def build_name_maps(model, arch, n_layers):
    """Return (gguf2hf, hf2gguf, norm_hf2gguf) built from the model's named parameters.

    norm_hf2gguf holds the 1-D (norm-gain) params that map to a GGUF tensor -- exactly the
    F32 continuous parameters the container keeps and distillation tunes."""
    from gguf import get_tensor_name_map
    enum = arch_to_enum(arch)
    if enum is None:
        raise ValueError(f"arch {arch!r} unknown to gguf; cannot build distill name maps")
    nm = get_tensor_name_map(enum, n_layers)
    gguf2hf, hf2gguf, norm = {}, {}, {}
    for name, p in model.named_parameters():
        base = name
        for suf in (".weight", ".bias"):
            if base.endswith(suf):
                stem, suffix = base[:-len(suf)], suf
                break
        else:
            stem, suffix = base, ""
        g = nm.get_name(stem)
        if g is None:
            continue
        gname = g + suffix
        hf2gguf[name] = gname
        gguf2hf[gname] = name
        if p.ndim == 1:
            norm[name] = gname
    return gguf2hf, hf2gguf, norm


def _dequant_gguf_tensor(t):
    """Dequantize ONE GGUF reader tensor -> f32 torch tensor (bit-faithful)."""
    import torch
    tt = t.tensor_type.name
    data = np.asarray(t.data)
    if tt in ("F32", "F16", "BF16"):
        w = data.astype(np.float32)
    elif tt in kq.DEQUANTIZE:
        ne0 = int(t.shape[0])
        w = kq.DEQUANTIZE[tt](data.reshape(data.shape[0], -1), ne0)
    else:
        raise ValueError(f"no decoder for {tt} ({t.name})")
    return torch.from_numpy(np.ascontiguousarray(w))


def dequant_state_dict(gguf_path, gguf2hf):
    """HF-named f32 state dict with weights dequantized from the GGUF (bit-faithful).

    Materializes the FULL f32 sd (~4 B/param) -- fine for small models and tests; big-model
    student construction uses stream_student_from_gguf instead (one tensor at a time)."""
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    sd = {}
    for t in r.tensors:
        hf = gguf2hf.get(t.name)
        if hf is None:
            raise KeyError(f"GGUF tensor {t.name} has no HF mapping (name-map gap)")
        sd[hf] = _dequant_gguf_tensor(t)
    return sd


def stream_student_from_gguf(model, gguf_path, gguf2hf, keep_f32=(), log=print):
    """Stream the GGUF into the (frozen) student skeleton's parameter storages IN PLACE.

    Per tensor: dequantize ONE GGUF tensor to f32, copy_ into the matching parameter (which
    casts to the skeleton's dtype/device), free -- peak extra RAM = one tensor (~2 GB at 12B)
    instead of the full f32 sd (~48 GB). Serves BOTH norm and scales student construction.

    keep_f32: iterable of HF param names whose EXACT f32 values (pre-cast) are captured and
    returned as {hf: f32 CPU tensor} -- the trainable norm gains initialize from these, so a
    bf16 skeleton does not round the norm init.

    STRICT ACCOUNTING (we no longer go through load_state_dict, so we enforce its `strict`
    contract ourselves):
      * every GGUF tensor MUST map to an existing parameter with a matching shape (KeyError /
        ValueError otherwise -- a GGUF tensor with no parameter home is a name-map gap);
      * every model parameter MUST be covered by some GGUF tensor, except rope/rotary
        position tables (buffers/params computed at init, permutation-independent).
    """
    import torch
    from gguf import GGUFReader
    params = dict(model.named_parameters())
    keep_f32 = set(keep_f32)
    captured = {}
    loaded = set()
    r = GGUFReader(gguf_path)
    with torch.no_grad():
        for t in r.tensors:
            hf = gguf2hf.get(t.name)
            if hf is None:
                raise KeyError(f"GGUF tensor {t.name} has no HF mapping (name-map gap)")
            p = params.get(hf)
            if p is None:
                raise KeyError(f"GGUF tensor {t.name} maps to {hf!r}, which is not a "
                               f"parameter of the student (strict accounting)")
            w = _dequant_gguf_tensor(t)
            if tuple(p.shape) != tuple(w.shape):
                raise ValueError(f"{t.name} -> {hf}: shape {tuple(w.shape)} != parameter "
                                 f"shape {tuple(p.shape)}")
            if hf in keep_f32:
                captured[hf] = w.clone()
            p.copy_(w)          # casts f32 -> skeleton dtype and moves to its device
            loaded.add(hf)
            del w
    not_loaded = [n for n in params if n not in loaded
                  and "rotary" not in n and "rope" not in n.lower()]
    assert not not_loaded, (f"{len(not_loaded)} parameter(s) not covered by any GGUF tensor "
                            f"(strict accounting): {not_loaded[:6]}")
    missing_f32 = keep_f32 - set(captured)
    assert not missing_f32, f"keep_f32 names absent from the GGUF: {sorted(missing_f32)[:6]}"
    log(f"streamed {len(loaded)} tensors into the student skeleton "
        f"(captured {len(captured)} f32 norm gains)")
    return captured


def _make_untied_student(model_dir, dtype=None, device_map=None):
    """Load the untied FP student skeleton.

    dtype: torch dtype for the skeleton weights (None -> float32). For 9B/12B a bf16
    skeleton keeps the frozen weights in ~half the RAM/VRAM; the trainable continuous
    params (d/dmin scales + norm gains) are held SEPARATELY in float32 by the callers and
    injected via functional_call, so bf16 here only affects the frozen backbone.
    device_map: passed through to from_pretrained ('auto'/'cuda' for a GPU-resident bf16
    skeleton); None keeps the current CPU behavior.
    """
    from transformers import AutoModelForCausalLM
    import torch
    if dtype is None:
        dtype = torch.float32
    kw = {"dtype": dtype, "tie_word_embeddings": False}
    if device_map is not None:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    # assert genuinely untied so we never clobber one role with the other on load
    if hasattr(model, "lm_head") and hasattr(model, "model") and \
       hasattr(model.model, "embed_tokens"):
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), \
            "student embeddings are tied; distillation would clobber embd/output"
    return model


def _kl_loss_fn(model, dev, forward):
    """Return kl_loss(f) computing KL(teacher topK + tail || student) for a cached chunk."""
    import torch

    def kl_loss(f):
        z = np.load(f)
        ids = torch.from_numpy(z["ids"]).long().to(dev)
        logits = forward(ids).float()
        slp = torch.log_softmax(logits, -1)
        ti = torch.from_numpy(z["top_i"]).long().to(dev)
        tlp = torch.from_numpy(z["top_lp"]).float().to(dev)
        tail = torch.from_numpy(z["tail"]).float().to(dev)
        s_top = slp.gather(-1, ti)
        p_top = tlp.exp()
        kl = (p_top * (tlp - s_top)).sum(-1)
        s_tail = torch.log1p((-s_top.exp().sum(-1)).clamp(min=-1 + 1e-7))
        kl = kl + tail.exp() * (tail - s_tail)
        return kl.mean()
    return kl_loss


def distill_norm(model_dir, arch, n_layers, gguf_path, teacher_dir, out_path,
                 steps=200, lr=3e-4, val_frac=0.1, dtype=None, device="cpu", log=print):
    """NORM: tune only the F32 norm gains; write them back into a copy of the GGUF.

    The frozen backbone loads in `dtype` (bf16 for 9B/12B to fit RAM/VRAM) and the GGUF is
    STREAMED into its parameter storages one tensor at a time (peak extra RAM = one tensor,
    not the full f32 sd). The trainable norm gains are SEPARATE float32 params (exact f32
    init captured during streaming) injected via a PARTIAL functional_call override, cast to
    the skeleton dtype only at forward time. Teacher targets are the precomputed top-K cache
    (no live teacher here)."""
    import torch
    from torch.func import functional_call
    if dtype is None:
        dtype = torch.float32
    dev = torch.device(device)
    files = sorted(_glob.glob(os.path.join(teacher_dir, "chunk*.npz")))
    assert files, teacher_dir
    n_val = max(1, int(len(files) * val_frac))
    val_files, train_files = files[:n_val], files[n_val:]

    model = _make_untied_student(model_dir, dtype=dtype)
    model.eval()
    model.to(dev)
    for p in model.parameters():
        p.requires_grad = False
    gguf2hf, hf2gguf, norm_map = build_name_maps(model, arch, n_layers)
    # stream the GGUF into the frozen skeleton (strict accounting inside); the exact f32
    # norm-gain values are captured pre-cast for the trainable init
    f32_norms = stream_student_from_gguf(model, gguf_path, gguf2hf,
                                         keep_f32=set(norm_map), log=log)
    norm_params = {hf: f32_norms[hf].to(dev, torch.float32).clone().requires_grad_(True)
                   for hf in norm_map}
    params = list(norm_params.values())
    log(f"trainable norm params: {sum(p.numel() for p in params)}")
    opt = torch.optim.Adam(params, lr=lr)

    def computed_sd():
        # PARTIAL override: only the norm gains; every other weight comes from the streamed
        # skeleton (functional_call falls back to module params for missing keys)
        return {hf: p.to(dtype) for hf, p in norm_params.items()}

    kl_loss = _kl_loss_fn(
        model, dev,
        lambda ids: functional_call(model, computed_sd(), (ids.unsqueeze(0),)).logits[0])

    def val():
        with torch.no_grad():
            return float(np.mean([kl_loss(f).item() for f in val_files]))

    def snapshot():
        return {hf: p.detach().clone() for hf, p in norm_params.items()}

    v0 = val()
    log(f"val KL before: {v0:.5f}")
    best = v0
    best_state = snapshot()
    rng = np.random.default_rng(0)
    for s in range(steps):
        f = train_files[rng.integers(len(train_files))]
        loss = kl_loss(f)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 25 == 0:
            v = val()
            log(f"step {s+1}/{steps} train {loss.item():.5f} val {v:.5f}")
            if v < best:
                best = v
                best_state = snapshot()
    log(f"val KL: {v0:.5f} -> {best:.5f} ({(best/v0-1)*100:+.1f}%)")

    replace_f32 = {gname: best_state[hf].cpu().numpy().astype(np.float32)
                   for hf, gname in norm_map.items()}
    reemit(gguf_path, out_path, replace_f32=replace_f32)
    json.dump({"val_before": v0, "val_after": best, "steps": steps, "lr": lr,
               "gguf": gguf_path}, open(out_path + ".normlog.json", "w"), indent=1)
    return {"val_before": v0, "val_after": best}


def distill_e3b(model_dir, arch, n_layers, gguf_path, teacher_dir, out_path,
                steps=300, lr=2e-4, device="cpu", val_frac=0.1, dtype=None, log=print):
    """E3B: tune every quantized tensor's fp16 d/dmin + all F32 norm gains.

    The functional_call skeleton loads the frozen backbone in `dtype` (bf16 for 9B/12B to fit
    RAM/VRAM) and the GGUF is STREAMED into its parameter storages (peak extra RAM = one
    tensor; no full f32 sd, no separate base_sd copy). The trainable d/dmin scales and norm
    gains stay FLOAT32; reconstruct()/norm injection cast to the skeleton dtype only at
    forward time via a PARTIAL functional_call override (non-quantized, non-norm tensors come
    from the streamed skeleton). Teacher targets are the precomputed top-K cache (no live
    teacher at distill time). NOTE the remaining at-scale memory term: the int16 A/B code
    consts (~4 B/param) -- see SCALE-NOTES (E3B deferred at 9B/12B pending streamed-A/B)."""
    import torch
    from torch.func import functional_call
    from gguf import GGUFReader
    if dtype is None:
        dtype = torch.float32

    files = sorted(_glob.glob(os.path.join(teacher_dir, "chunk*.npz")))
    assert files, teacher_dir
    n_val = max(1, int(len(files) * val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    dev = torch.device(device)

    model = _make_untied_student(model_dir, dtype=dtype)
    model.eval()
    model.to(dev)
    for p in model.parameters():
        p.requires_grad = False
    gguf2hf, hf2gguf, norm_map = build_name_maps(model, arch, n_layers)

    # stream the GGUF into the frozen skeleton (strict accounting inside); exact f32 norm init
    f32_norms = stream_student_from_gguf(model, gguf_path, gguf2hf,
                                         keep_f32=set(norm_map), log=log)
    entries = scale_tune.build_torch_params(gguf_path)
    params = []
    for name, e in entries.items():
        e["d"] = e["d"].to(dev).requires_grad_(True)
        params.append(e["d"])
        if e["dmin"] is not None:
            e["dmin"] = e["dmin"].to(dev).requires_grad_(True)
            params.append(e["dmin"])
        e["A"] = e["A"].to(dev)
        if e["B"] is not None:
            e["B"] = e["B"].to(dev)
    norm_params = {}
    for hf in norm_map:
        norm_params[hf] = f32_norms[hf].to(dev, torch.float32).clone().requires_grad_(True)
        params.append(norm_params[hf])
    n_train = sum(p.numel() for p in params)
    log(f"trainable params: {n_train} ({len(entries)} quant tensors + {len(norm_params)} norms)")

    def computed_sd():
        # PARTIAL override: reconstructed quant tensors + norm gains; everything else falls
        # back to the streamed skeleton parameters
        sd = {gguf2hf[gname]: scale_tune.reconstruct(e, dtype=dtype)
              for gname, e in entries.items()}
        for hf, p in norm_params.items():
            sd[hf] = p.to(dtype)
        return sd

    kl_loss = _kl_loss_fn(
        model, dev,
        lambda ids: functional_call(model, computed_sd(), (ids.unsqueeze(0),)).logits[0])

    def val():
        with torch.no_grad():
            return float(np.mean([kl_loss(f).item() for f in val_files]))

    def snapshot():
        return {"entries": {n: (e["d"].detach().clone(),
                                e["dmin"].detach().clone() if e["dmin"] is not None else None)
                            for n, e in entries.items()},
                "norms": {n: p.detach().clone() for n, p in norm_params.items()}}

    opt = torch.optim.Adam(params, lr=lr)
    v0 = val()
    log(f"val KL before: {v0:.5f}")
    best = v0
    best_state = snapshot()
    rng = np.random.default_rng(0)
    for s in range(steps):
        f = train_files[rng.integers(len(train_files))]
        loss = kl_loss(f)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 25 == 0:
            v = val()
            log(f"step {s+1}/{steps} train {loss.item():.5f} val {v:.5f}")
            if v < best:
                best = v
                best_state = snapshot()
    log(f"val KL: {v0:.5f} -> {best:.5f} ({(best/v0-1)*100:+.1f}%)")

    r = GGUFReader(gguf_path)
    replace, replace_f32 = {}, {}
    for t in r.tensors:
        if t.name in entries:
            d_new, dm_new = best_state["entries"][t.name]
            raw = np.asarray(t.data)
            new = scale_tune.write_back(raw, entries[t.name]["ttype"],
                                        d_new.cpu().numpy(),
                                        dm_new.cpu().numpy() if dm_new is not None else None)
            replace[t.name] = new.reshape(-1)
    for hf, gname in norm_map.items():
        replace_f32[gname] = best_state["norms"][hf].cpu().numpy().astype(np.float32)
    reemit(gguf_path, out_path, replace=replace, replace_f32=replace_f32)
    json.dump({"val_before": v0, "val_after": best, "steps": steps, "lr": lr,
               "n_trainable": n_train, "gguf": gguf_path},
              open(out_path + ".e3blog.json", "w"), indent=1)
    return {"val_before": v0, "val_after": best}
