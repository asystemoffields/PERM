#!/usr/bin/env python3
# =============================================================================
#  ##  CODE REVIEWED (Fable, 2026-07-04) — G3 GATE ON REAL WEIGHTS STILL REQUIRED  ##
#  Review applied 3 fixes: language_model prefix handling, attn_output input_perm
#  composition (the permef-v1 bug class), multimodal guard (gemma4). The
#  acknowledge_unreviewed flag stays until g3_check passes on the real checkpoint.
#  Fable DERIVATION (2026-07-04) implemented by Opus from the tensor-edit table in
#  src/spacemaps/gemma4.py. NOT reviewed against the derivation; NO G3 logits-equality
#  gate run on real Gemma-4 weights. Every entry point raises NotImplementedError unless
#  acknowledge_unreviewed=True. Do NOT ship an artifact built with this until Fable signs
#  off AND g3_check returns rel < 1e-4 on the real (text-stack, and if -it/vision ships,
#  multimodal) checkpoint.
# =============================================================================
"""Space map: Gemma4 dense text stack (arch gemma4; google/gemma-4-12b[-it]).

Model shape (12B): d=3840, ffn=15360, 48 layers (40 sliding: head_dim 256 / 8 kv-heads;
8 global at 5,11,...,47: head_dim 512 / 1 kv-head, K=V weight sharing), TIED embeddings
(no lm_head tensor), vocab 262144, final logit softcap 30.0.

Implemented (first-ship subset): P_res [3840], P_ffn [15360] (all 48), P_vo [256] (SLIDING
layers only, 40 of 48). GLOBAL layers frozen identity in v1 (K=V sharing + proportional
RoPE make a legal perm exist but with three coupled subtleties -- bad trade for v1). Tied
embeddings: the residual perm carries the 262K x 3840 embedding win in BOTH roles.

Gemma norms are PLAIN-weight RMSNorm (init 1, no +1 offset) -> permuting commutes exactly.
Scalar embed*sqrt(d), per-layer layer_scalar, tanh logit softcap all commute with P.
Global layers HAVE NO v_proj (attention_k_eq_v). Sliding v_norm is per-head RMS with NO
weight -> nothing to co-permute in P_vo.

MULTIMODAL: checkpoint nests the text stack (model.language_model.*). Any projector that
writes into the 3840 residual must have its out-rows (+ any 3840 projector norm) permuted
by P; a text-only G3 will NOT catch it -- run a multimodal smoke or ship text-only with a
documented caveat.
"""
import numpy as np

MODEL_TYPE = "gemma4"
ARCH = "gemma4"

_HD = 256  # sliding-layer head_dim


def _guard(acknowledge_unreviewed):
    if not acknowledge_unreviewed:
        raise NotImplementedError(
            "gemma4 spacemap is PENDING FABLE REVIEW + a G3 gate on real weights; "
            "pass acknowledge_unreviewed=True to run it anyway (do not ship the result "
            "until Fable signs off and g3_check rel < 1e-4 on the real checkpoint).")


def dims_from_config(config, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    txt = getattr(config, "text_config", config)
    n_layers = int(txt.num_hidden_layers)
    global_layers = []
    lt = getattr(txt, "layer_types", None)
    if lt:
        global_layers = [i for i, t in enumerate(lt)
                         if "global" in str(t).lower() or "full" in str(t).lower()]
    else:
        interval = getattr(txt, "sliding_window_pattern", 6)
        global_layers = [i for i in range(n_layers) if (i + 1) % interval == 0]
    sliding = [i for i in range(n_layers) if i not in set(global_layers)]
    return {
        "d_model": int(txt.hidden_size),
        "d_ffn": int(txt.intermediate_size),
        "n_layers": n_layers,
        "n_kv": int(getattr(txt, "num_key_value_heads", 8)),
        "n_heads": int(getattr(txt, "num_attention_heads", 16)),
        "head_dim": _HD,
        "sliding_layers": sliding,
        "global_layers": global_layers,
    }


def identity_perms(dims, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    return {
        "res": np.arange(dims["d_model"]),
        "ffn": [np.arange(dims["d_ffn"]) for _ in range(dims["n_layers"])],
        "vo": {l: [np.arange(_HD) for _ in range(dims["n_kv"])] for l in dims["sliding_layers"]},
    }


def input_perm(gguf_name, perms, dims=None, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    if gguf_name in ("token_embd.weight", "output.weight"):
        return np.asarray(perms["res"])
    if not gguf_name.startswith("blk."):
        return None
    parts = gguf_name.split(".")
    layer, kind = int(parts[1]), parts[2]
    if kind in ("attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"):
        return np.asarray(perms["res"])
    if kind == "ffn_down":
        return np.asarray(perms["ffn"][layer])
    if kind == "attn_output":
        # sliding layers: o ne[0] = 16*256 with P_vo per kv-head; global layers: 16*512,
        # frozen identity in v1 (K=V sharing analysis in the derivation doc).
        assert dims is not None, "dims required for attn_output input_perm"
        if layer in perms.get("vo", {}):
            idx = np.arange(dims["n_heads"] * _HD)
            group = dims["n_heads"] // dims["n_kv"]
            for h in range(dims["n_kv"]):
                p = np.asarray(perms["vo"][layer][h])
                for qh in range(group * h, group * h + group):
                    idx[qh * _HD:(qh + 1) * _HD] = qh * _HD + p
            return idx
        return None  # global layer: identity (frozen)
    return None


def _prefix(sd):
    """Gemma multimodal checkpoints nest the text stack under model.language_model.*."""
    if any(k.startswith("model.language_model.") for k in sd):
        return "model.language_model."
    return "model."


def apply_perms(sd, perms, dims, consume=False, acknowledge_unreviewed=False,
                strip_vision=False):
    """Apply P_res + P_ffn + P_vo(sliding) to a Gemma4 HF state dict (text stack).

    strip_vision=True: DELETE vision-tower/projector tensors and proceed text-only
    (they are dropped by text-only GGUF conversion anyway). The resulting artifact is
    a TEXT-ONLY ship; multimodal use would need the projector out-row permutation
    (see the derivation doc)."""
    _guard(acknowledge_unreviewed)
    import torch
    P = np.asarray(perms["res"])
    pre_root = _prefix(sd)
    # Multimodal guard: the vision tower passes through untouched, but any projector
    # writing INTO the permuted text residual would silently break vision. Text-only
    # shipping is the reviewed path; acknowledge explicitly to proceed with vision
    # tensors present (their projector output rows are NOT permuted here).
    vision_keys = [k for k in sd if "vision" in k or "multi_modal_projector" in k]
    if vision_keys and strip_vision:
        for k in vision_keys:
            sd.pop(k)
        print(f"[gemma4 spacemap] stripped {len(vision_keys)} vision tensors (text-only ship)")
        vision_keys = []
    if vision_keys:
        raise NotImplementedError(
            f"checkpoint has {len(vision_keys)} vision-tower tensors; the gemma4 spacemap "
            "is reviewed for TEXT-ONLY use (the multimodal projector seam is unhandled — "
            "see src/spacemaps/gemma4.py derivation). Strip the vision tower or extend "
            "the map with the projector out-row permutation first.")

    def sel(w, dim, idx):
        return w.index_select(dim, torch.from_numpy(np.ascontiguousarray(idx)).long())

    def take(name):
        return sd.pop(name) if consume else sd[name]

    out = {name: w for name, w in list(sd.items())}
    emb = pre_root + "embed_tokens.weight"
    out[emb] = sel(take(emb), 1, P)   # tied: this IS the lm_head; one edit covers both roles
    nrm = pre_root + "norm.weight"
    out[nrm] = sel(take(nrm), 0, P)

    group = dims["n_heads"] // dims["n_kv"]
    sliding = set(dims["sliding_layers"])
    for l in range(dims["n_layers"]):
        pre = f"{pre_root}layers.{l}."
        Pf = np.asarray(perms["ffn"][l])
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight",
                  "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"):
            if pre + n in sd:
                out[pre + n] = sel(take(pre + n), 0, P)
        out[pre + "mlp.gate_proj.weight"] = sel(sel(take(pre + "mlp.gate_proj.weight"), 1, P), 0, Pf)
        out[pre + "mlp.up_proj.weight"] = sel(sel(take(pre + "mlp.up_proj.weight"), 1, P), 0, Pf)
        out[pre + "mlp.down_proj.weight"] = sel(sel(take(pre + "mlp.down_proj.weight"), 0, P), 1, Pf)
        out[pre + "self_attn.q_proj.weight"] = sel(take(pre + "self_attn.q_proj.weight"), 1, P)
        out[pre + "self_attn.k_proj.weight"] = sel(take(pre + "self_attn.k_proj.weight"), 1, P)
        o = sel(take(pre + "self_attn.o_proj.weight"), 0, P)
        if l in sliding:
            v = sel(take(pre + "self_attn.v_proj.weight"), 1, P)   # sliding layers HAVE v_proj
            for h in range(dims["n_kv"]):
                p = np.asarray(perms["vo"][l][h])
                v_rows = np.arange(v.shape[0])
                v_rows[h * _HD:(h + 1) * _HD] = h * _HD + p
                v = sel(v, 0, v_rows)
                for qh in [group * h + j for j in range(group)]:
                    o_cols = np.arange(o.shape[1])
                    o_cols[qh * _HD:(qh + 1) * _HD] = qh * _HD + p
                    o = sel(o, 1, o_cols)
            out[pre + "self_attn.v_proj.weight"] = v
        # global layers: no v_proj (K=V share k_proj), P_vo frozen identity in v1
        out[pre + "self_attn.o_proj.weight"] = o
    return out


# ---------------------------------------------------------------- save/load perms

def save_perms(perms, path, dims=None, acknowledge_unreviewed=False):
    """npz layout: res [d_model], ffn [n_layers, d_ffn], vo [n_vo_layers, n_kv, head_dim]
    with a parallel vo_layers index (P_vo lives on SLIDING layers only -> vo is a DICT keyed
    by layer; global layers are frozen identity in v1 and absent from vo)."""
    _guard(acknowledge_unreviewed)
    vo = perms["vo"]
    vo_layers = sorted(int(l) for l in vo)
    if vo_layers:
        vo_arr = np.stack([np.stack([np.asarray(p) for p in vo[l]]) for l in vo_layers])
    else:
        vo_arr = np.zeros((0, dims["n_kv"] if dims else 0, _HD), dtype=np.int64)
    np.savez_compressed(
        path,
        res=np.asarray(perms["res"]),
        ffn=np.stack([np.asarray(p) for p in perms["ffn"]]),
        vo=vo_arr,
        vo_layers=np.asarray(vo_layers, dtype=np.int64),
    )


def load_perms(path, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    z = np.load(path)
    n_layers = z["ffn"].shape[0]
    vo_layers = [int(l) for l in z["vo_layers"]]
    n_kv = z["vo"].shape[1] if z["vo"].size else 0
    vo = {l: [z["vo"][i, h] for h in range(n_kv)] for i, l in enumerate(vo_layers)}
    return {
        "res": z["res"],
        "ffn": [z["ffn"][l] for l in range(n_layers)],
        "vo": vo,
    }


# ---------------------------------------------------------------- optimize

def _res_space_tensors(weights, dims):
    # tied embeddings: token_embd (and output.weight if the conversion emits one) carry P_res.
    names = ["token_embd.weight", "output.weight"]
    sliding = set(dims["sliding_layers"])
    for l in range(dims["n_layers"]):
        names += [f"blk.{l}.attn_q.weight", f"blk.{l}.attn_k.weight"]
        if l in sliding:
            names.append(f"blk.{l}.attn_v.weight")   # global layers have no v_proj (K=V share)
        names += [f"blk.{l}.ffn_gate.weight", f"blk.{l}.ffn_up.weight"]
    return [n for n in names if n in weights]


def optimize(weights, ttypes, qws, dims, rows_sample=16384, log=print,
             acknowledge_unreviewed=False):
    """Choose P_res + P_ffn(all layers) + P_vo(sliding layers) minimizing the container
    objective. Global layers are frozen identity in v1 (K=V sharing + proportional RoPE make a
    legal perm exist but with coupled subtleties -- see the derivation). Returns (perms,
    report). vo is a DICT keyed by sliding layer."""
    _guard(acknowledge_unreviewed)
    import time
    from ..objective import tensor_err, col_log_rms
    t0 = time.time()

    def res_objective(perm):
        total = 0.0
        for n in _res_space_tensors(weights, dims):
            total += tensor_err(weights[n], ttypes[n], qws.get(n), perm=perm,
                                rows_sample=rows_sample)
        return total

    def build_res_keys():
        profs = [(n, col_log_rms(weights[n])) for n in _res_space_tensors(weights, dims)]
        stand = [(p - p.mean()) / (p.std() + 1e-9) for _, p in profs]
        keys = {"mean-logrms": np.mean(stand, axis=0)}
        emb = [p for n, p in profs if n == "token_embd.weight"]
        if emb:
            keys["embd-logrms"] = emb[0]
        blk = [s for (n, _), s in zip(profs, stand) if n.startswith("blk.")]
        if blk:
            keys["blk-logrms"] = np.mean(blk, axis=0)
        return keys

    base = res_objective(None)
    res_perm, res_val, res_name = np.arange(dims["d_model"]), base, "identity"
    for kname, key in build_res_keys().items():
        perm = np.argsort(key, kind="stable")
        val = res_objective(perm)
        log(f"  res sort[{kname}]: {val:.6g} ({(val/base-1)*100:+.2f}%)")
        if val < res_val:
            res_perm, res_val, res_name = perm, val, kname
    log(f"  res -> {res_name} ({(res_val/base-1)*100:+.2f}%)")

    ffn_perms, ffn_val, ffn_base = [], 0.0, 0.0
    vo_perms, vo_val, vo_base = {}, 0.0, 0.0
    hd, nkv, nh = dims["head_dim"], dims["n_kv"], dims["n_heads"]
    group = nh // nkv
    sliding = set(dims["sliding_layers"])
    for l in range(dims["n_layers"]):
        fn = f"blk.{l}.ffn_down.weight"
        if fn in weights:
            fb = tensor_err(weights[fn], ttypes[fn], qws.get(fn))
            fp = np.argsort(col_log_rms(weights[fn]), kind="stable")
            fv = tensor_err(weights[fn], ttypes[fn], qws.get(fn), perm=fp)
            if fv >= fb:
                fp, fv = np.arange(dims["d_ffn"]), fb
            ffn_perms.append(fp)
            ffn_val += fv
            ffn_base += fb
        else:
            ffn_perms.append(np.arange(dims["d_ffn"]))
        on = f"blk.{l}.attn_output.weight"
        if l in sliding and on in weights:
            x = np.asarray(weights[on])
            ob = tensor_err(x, ttypes[on], qws.get(on))
            heads = []
            for h in range(nkv):
                qs = [q for q in range(nh) if q // group == h]
                key = np.zeros(hd)
                for q in qs:
                    key = key + col_log_rms(x[:, q * hd:(q + 1) * hd])
                heads.append(np.argsort(key, kind="stable"))
            full_idx = np.arange(nh * hd)
            for q in range(nh):
                full_idx[q * hd:(q + 1) * hd] = q * hd + heads[q // group]
            ov = tensor_err(x, ttypes[on], qws.get(on), perm=full_idx)
            if ov >= ob:
                heads = [np.arange(hd) for _ in range(nkv)]
                ov = ob
            vo_perms[l] = heads
            vo_val += ov
            vo_base += ob

    perms = {"res": res_perm, "ffn": ffn_perms, "vo": vo_perms}
    report = {
        "res": {"base": base, "opt": res_val, "rel": res_val / base - 1 if base else 0.0},
        "ffn": {"base": ffn_base, "opt": ffn_val,
                "rel": ffn_val / ffn_base - 1 if ffn_base else 0.0},
        "vo": {"base": vo_base, "opt": vo_val,
               "rel": vo_val / vo_base - 1 if vo_base else 0.0},
        "rows_sample": rows_sample, "seconds": time.time() - t0,
        "note": "global layers frozen identity (v1); P_vo sliding layers only",
    }
    return perms, report


# ---------------------------------------------------------------- imatrix permute

def permute_imatrix(src, dst, perms, dims, acknowledge_unreviewed=False):
    """Copy an imatrix GGUF and permute every recognized .in_sum2 along ne[0] to match the
    perms. Global-layer attn_output (frozen identity in v1) yields input_perm None and is left
    unchanged."""
    _guard(acknowledge_unreviewed)
    import shutil
    from gguf import GGUFReader
    shutil.copyfile(src, dst)
    r = GGUFReader(dst, "r+")
    n_patched = 0
    for t in r.tensors:
        if not t.name.endswith(".in_sum2"):
            continue
        base = t.name[:-len(".in_sum2")]
        idx = input_perm(base, perms, dims, acknowledge_unreviewed=acknowledge_unreviewed)
        if idx is None:
            continue
        v = t.data
        assert v.ndim == 1, (t.name, v.shape)
        v[...] = v[np.asarray(idx)]
        n_patched += 1
    del r
    return n_patched


# ---------------------------------------------------------------- G3 gate

def g3_check(model, perms, dims, ids, acknowledge_unreviewed=False):
    """Logits-equality oracle (text stack). Reference forward, apply the permutation by
    writing the permuted state dict back (load_state_dict strict=False; re-tie the tied
    embeddings), re-run. Returns (max|dlogit|, rel). Mutates `model` in place.

    TEXT-ONLY: a multimodal projector writing into the residual is NOT exercised here (ship
    text-only or add a multimodal smoke -- see the module docstring)."""
    _guard(acknowledge_unreviewed)
    import torch
    model.eval()
    with torch.no_grad():
        ref = model(ids).logits.clone()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", True))
    out = apply_perms(sd, perms, dims, consume=True, strip_vision=True,
                      acknowledge_unreviewed=acknowledge_unreviewed)
    if tied:
        out.pop("lm_head.weight", None)   # tied: leave lm_head sharing the permuted embd
    model.load_state_dict(out, strict=False)
    if tied and hasattr(model, "tie_weights"):
        model.tie_weights()
    with torch.no_grad():
        new = model(ids).logits
    d = (ref - new).abs().max().item()
    rel = d / ref.abs().max().item()
    return d, rel
