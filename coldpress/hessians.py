#!/usr/bin/env python3
"""Generic input-Hessian collector for EF (GPTQ-style error feedback).

For every matmul we will EF-encode, we need the input second-moment H = X^T X (f32),
where X are the activations flowing INTO that matmul over calibration text. This module
hooks the FP teacher and accumulates those Hessians, keyed by the GGUF tensor name the
matmul maps to -- so ef.py can look a Hessian up directly by the name it reads from the
GGUF, with no per-architecture translation table.

  [FABLE-REVIEW] The name-mapping seam. We map each nn.Linear's module name to a GGUF
  tensor name via gguf's TensorNameMap (gguf.get_tensor_name_map(arch_enum, n_layers)),
  the SAME map convert_hf_to_gguf uses. A wrong mapping would silently mis-key a Hessian.
  Mitigations, both enforced here:
    * shape cross-check: H.shape[0] (== module in_features) MUST equal the GGUF tensor's
      ne[0]; a mismatch is a hard error (never a silent mis-key).
    * unmapped modules (no GGUF name, or a name absent from the GGUF) are logged and
      skipped, never silently accumulated under a guessed key.

Layer sharding: pass layer_lo/layer_hi to hook only blocks in [lo, hi); non-block tensors
(token_embd, output, output_norm) are included only in the shard that starts at layer 0.
Big models => run several disjoint shards; each writes its own per-tensor npz files.

Saved layout (new, canonical): one npz per tensor at <outdir>/<gguf_name with . -> _>.npz
holding H (f32 [d,d]) and n (token count). load_hessian() also reads the OLD family-keyed
layout (blk.L.qkv_in.npz etc.) for backward compatibility.

  token_embd.weight has no input Hessian (it is a lookup table, not a matmul). EF upgrades
  it with informed column weights derived from the layer-0 attention input Hessian instead;
  this collector simply never emits a token_embd Hessian.
"""
import os

import numpy as np


# Family fallbacks: EF asks for a Hessian by GGUF tensor name; if the per-tensor file is
# absent we fall back to the shared-input family key of the OLD layout. q/k/v share one
# input; gate/up share one input.
_FAMILY_FALLBACK = {
    "attn_q": "qkv_in", "attn_k": "qkv_in", "attn_v": "qkv_in",
    "attn_output": "o_in",
    "ffn_gate": "gateup_in", "ffn_up": "gateup_in",
    "ffn_down": "down_in",
}


def arch_to_enum(arch):
    """Map a GGUF arch string (e.g. 'qwen3') to gguf.MODEL_ARCH enum, or None."""
    import gguf
    inv = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}
    return inv.get(arch)


def _block_index(gguf_name):
    """Block index for a GGUF tensor name, or -1 for non-block (embd/output/final norm)."""
    if gguf_name.startswith("blk."):
        try:
            return int(gguf_name.split(".")[1])
        except (IndexError, ValueError):
            return -1
    return -1


def build_name_map(model, arch, n_layers):
    """{module_name: gguf_base_name} for every nn.Linear/nn.Embedding, via TensorNameMap.

    gguf_base_name has NO '.weight' suffix (that is appended by callers). Modules that do
    not map are omitted (callers log them)."""
    import torch.nn as nn
    from gguf import get_tensor_name_map
    enum = arch_to_enum(arch)
    if enum is None:
        raise ValueError(f"arch {arch!r} is unknown to the installed gguf package; "
                         f"cannot map module names to GGUF tensor names")
    nm = get_tensor_name_map(enum, n_layers)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            g = nm.get_name(name)
            if g is not None:
                out[name] = g
    return out


def collect_hessians(model, batches, gguf_ne0, arch, n_layers,
                     layer_lo=0, layer_hi=None, progress_every=16, log=print):
    """Accumulate X^T X for every hooked matmul in the layer shard [layer_lo, layer_hi).

    model      : an FP nn.Module (eval mode) whose forward we run over `batches`.
    batches    : iterable of LongTensor token id sequences (each [T] or [1, T]).
    gguf_ne0   : {gguf_tensor_name(with .weight): ne0} from the target GGUF, for the
                 shape cross-check. A module whose GGUF name is not present here is
                 treated as unmapped (logged, skipped).
    arch       : GGUF arch string (e.g. 'qwen3').
    n_layers   : block count (for the TensorNameMap).

    Returns (hessians, unmapped) where hessians = {gguf_name(with .weight):
    {"H": f32 [d,d], "n": int}} and unmapped = list of skipped module names.
    """
    import torch
    import torch.nn as nn
    if layer_hi is None:
        layer_hi = n_layers
    name_map = build_name_map(model, arch, n_layers)

    acc, cnt = {}, {}
    unmapped, mismatched, plan = [], [], []

    # Plan first, and REFUSE mismatches BEFORE registering any hook -- so a shape mismatch
    # never leaves hooks dangling on the (possibly shared) model.
    for mname, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        base = name_map.get(mname)
        if base is None:
            unmapped.append(mname)
            continue
        gname = base + ".weight"
        if gname not in gguf_ne0:
            unmapped.append(mname + f" (->{gname}, absent from GGUF)")
            continue
        blk = _block_index(gname)
        in_shard = (layer_lo <= blk < layer_hi) if blk >= 0 else (layer_lo == 0)
        if not in_shard:
            continue
        if isinstance(mod, nn.Linear) and mod.in_features != gguf_ne0[gname]:
            mismatched.append((mname, gname, mod.in_features, gguf_ne0[gname]))
            continue
        plan.append((mod, gname))

    if mismatched:
        for mname, gname, inf, ne0 in mismatched:
            log(f"  SHAPE MISMATCH {mname} -> {gname}: in_features={inf} ne0={ne0}")
        raise ValueError(f"{len(mismatched)} module(s) map to a GGUF tensor with a "
                         f"different ne[0]; refusing to mis-key a Hessian (see log above)")

    def make_hook(gname):
        def fn(mod, inp, out):
            x = inp[0]
            if not torch.is_floating_point(x):
                return  # Embedding lookup role: ids in, no meaningful input Hessian
            x = x.detach().reshape(-1, x.shape[-1]).float()
            xtx = (x.T @ x).clone()
            if gname not in acc:
                acc[gname] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32)
                cnt[gname] = 0
            acc[gname] += xtx
            cnt[gname] += x.shape[0]
        return fn

    handles = [mod.register_forward_hook(make_hook(gname)) for mod, gname in plan]
    log(f"hooked {len(handles)} matmuls in layers [{layer_lo},{layer_hi}); "
        f"{len(unmapped)} module(s) unmapped/skipped")
    try:
        with torch.inference_mode():
            for c, ids in enumerate(batches):
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                model(ids)
                if progress_every and (c + 1) % progress_every == 0:
                    log(f"  chunk {c + 1}")
    finally:
        for h in handles:
            h.remove()

    hessians = {}
    for gname, H in acc.items():
        Hn = H.numpy().astype(np.float32)
        # final shape cross-check against the GGUF (defense in depth)
        if gname in gguf_ne0 and Hn.shape[0] != gguf_ne0[gname]:
            raise ValueError(f"{gname}: collected H.shape[0]={Hn.shape[0]} != "
                             f"ne0={gguf_ne0[gname]}")
        hessians[gname] = {"H": Hn, "n": int(cnt[gname])}
    return hessians, unmapped


def save_hessians(hessians, outdir):
    os.makedirs(outdir, exist_ok=True)
    for gname, h in hessians.items():
        fn = os.path.join(outdir, gname.replace(".", "_") + ".npz")
        np.savez_compressed(fn, H=h["H"], n=h["n"], name=gname)


def _fallback_key(gguf_name):
    """Old-layout family key for a GGUF tensor name, e.g. blk.3.attn_q.weight ->
    blk.3.qkv_in ; output.weight -> lmhead_in. None if no fallback."""
    if gguf_name == "output.weight":
        return "lmhead_in"
    if not gguf_name.startswith("blk."):
        return None
    parts = gguf_name.split(".")
    if len(parts) < 3:
        return None
    layer, kind = parts[1], parts[2]
    fam = _FAMILY_FALLBACK.get(kind)
    return f"blk.{layer}.{fam}" if fam else None


def load_hessian(hessdir, gguf_name):
    """Load a Hessian for a GGUF tensor name. Prefers the new per-tensor file; falls back
    to the OLD family-keyed layout (blk.L.qkv_in.npz, ..., lmhead_in.npz). Returns
    (H f64 [d,d], n int) or raises FileNotFoundError."""
    cand = [os.path.join(hessdir, gguf_name.replace(".", "_") + ".npz")]
    fk = _fallback_key(gguf_name)
    if fk is not None:
        cand.append(os.path.join(hessdir, fk.replace(".", "_") + ".npz"))
    for path in cand:
        if os.path.exists(path):
            z = np.load(path)
            return z["H"].astype(np.float64), int(z["n"])
    raise FileNotFoundError(f"no Hessian for {gguf_name} (tried {cand})")


def has_hessian(hessdir, gguf_name):
    try:
        load_hessian(hessdir, gguf_name)
        return True
    except FileNotFoundError:
        return False
