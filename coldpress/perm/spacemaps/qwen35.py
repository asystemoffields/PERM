#!/usr/bin/env python3
# =============================================================================
#  ##  CODE REVIEWED (Fable, 2026-07-04) — G3 GATE ON REAL WEIGHTS STILL REQUIRED  ##
#  Review applied 3 fixes: language_model prefix handling, attn_output input_perm
#  composition (the permef-v1 bug class), multimodal guard (gemma4). The
#  acknowledge_unreviewed flag stays until g3_check passes on the real checkpoint.
#  This space map is a Fable DERIVATION (2026-07-04) implemented by Opus from the
#  tensor-edit table in src/spacemaps/qwen35.py. It has NOT been reviewed against the
#  original derivation, and NO G3 logits-equality gate has been run on real Qwen3.5
#  weights. Every entry point raises NotImplementedError unless you pass
#  acknowledge_unreviewed=True. Do NOT ship an artifact built with this until Fable
#  signs off AND g3_check returns rel < 1e-4 on the real checkpoint.
# =============================================================================
"""Space map: Qwen3.5 dense hybrid (arch qwen35; derived for Qwen/Qwen3.5-9B).

Model shape (9B): d=4096, ffn=12288, 32 layers (24 linear-attn + 8 full-attn at
3,7,...,31), full-attn 16q/4kv x head_dim 256, UNTIED lm_head, vocab 248320.

Implemented here (first-ship subset, per src table 'IMPLEMENTATION ORDER'):
  P_res [4096], P_ffn [12288] (all 32 layers), P_vo [256] (full-attn layers only).
  P_lav (linear-attn v-head space) frozen identity in v1.

THE GATE TRAP (src table P_vo): q_proj is DOUBLED per full-attn q-head q: rows
[q*512:q*512+256] = query (MUST NOT permute -- shared q_norm + RoPE), rows
[q*512+256:(q+1)*512] = gate (sigmoid-multiplies the attn output CHANNEL-FOR-CHANNEL and
MUST co-permute with V->O). GQA: q-heads {4h..4h+3} share kv-head h.
"""
import numpy as np

MODEL_TYPE = "qwen35"
ARCH = "qwen35"

_HD = 256          # full-attn head_dim
_QHEAD = 512       # doubled q-head width (query 256 + gate 256)


def _guard(acknowledge_unreviewed):
    if not acknowledge_unreviewed:
        raise NotImplementedError(
            "qwen35 spacemap is PENDING FABLE REVIEW + a G3 gate on real weights; "
            "pass acknowledge_unreviewed=True to run it anyway (do not ship the result "
            "until Fable signs off and g3_check rel < 1e-4 on the real checkpoint).")


def dims_from_config(config, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    # multimodal wrapper config nests the text stack under text_config; a plain text config
    # is used directly (mirrors the gemma4 map + the onboard text-stack handling).
    txt = getattr(config, "text_config", config)
    n_layers = int(txt.num_hidden_layers)
    # full-attn layer indices: prefer an explicit layer_types list; else the 3,7,... pattern
    full = []
    lt = getattr(txt, "layer_types", None)
    if lt:
        full = [i for i, t in enumerate(lt) if "full" in str(t).lower()]
    else:
        interval = getattr(txt, "full_attention_interval", 4)
        full = [i for i in range(n_layers) if (i + 1) % interval == 0]
    return {
        "d_model": int(txt.hidden_size),
        "d_ffn": int(txt.intermediate_size),
        "n_layers": n_layers,
        "n_kv": int(getattr(txt, "num_key_value_heads", 4)),
        "n_heads": int(getattr(txt, "num_attention_heads", 16)),
        "head_dim": _HD,
        "full_attn_layers": full,
    }


def identity_perms(dims, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    full = dims["full_attn_layers"]
    return {
        "res": np.arange(dims["d_model"]),
        "ffn": [np.arange(dims["d_ffn"]) for _ in range(dims["n_layers"])],
        "vo": {l: [np.arange(_HD) for _ in range(dims["n_kv"])] for l in full},
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
        # o ne[0] = n_heads*head_dim; q-heads {group*h..group*h+group-1} carry P_vo[l][h].
        # Linear-attn layers (P_lav frozen) and layers without vo entries -> identity.
        assert dims is not None, "dims required for attn_output input_perm"
        n_cols = dims["n_heads"] * _HD
        idx = np.arange(n_cols)
        if layer in perms.get("vo", {}):
            group = dims["n_heads"] // dims["n_kv"]
            for h in range(dims["n_kv"]):
                p = np.asarray(perms["vo"][layer][h])
                for qh in range(group * h, group * h + group):
                    idx[qh * _HD:(qh + 1) * _HD] = qh * _HD + p
        return idx
    return None


def _prefix(sd):
    """Qwen3.5 checkpoints nest the text stack under model.language_model.* (verified in
    the fetched Qwen/Qwen3.5-9B safetensors index); plain model.* also accepted."""
    if any(k.startswith("model.language_model.") for k in sd):
        return "model.language_model."
    return "model."


def apply_perms(sd, perms, dims, consume=False, acknowledge_unreviewed=False,
                strip_vision=False):
    """Apply P_res + P_ffn + P_vo(full-attn) to a Qwen3.5 HF state dict (text stack names).

    Mirrors the qwen3 pattern; the ONLY structural difference is the doubled q_proj (the
    gate half co-permutes, the query half does not) and the per-layer full/linear split.

    strip_vision: accepted for CLI signature uniformity but a NO-OP here. Qwen3.5 multimodal
    checkpoints keep the text stack under model.language_model.*; any vision-tower tensors are
    left untouched in the state dict and are dropped by the text-only GGUF conversion (unlike
    gemma4, which deletes them explicitly). Extending qwen35 to a real strip/multimodal seam
    is a reviewed follow-up.
    """
    _guard(acknowledge_unreviewed)
    import torch
    P = np.asarray(perms["res"])
    pre_root = _prefix(sd)

    def sel(w, dim, idx):
        return w.index_select(dim, torch.from_numpy(np.ascontiguousarray(idx)).long())

    def take(name):
        return sd.pop(name) if consume else sd[name]

    out = {name: w for name, w in list(sd.items())}
    Pt = P
    emb = pre_root + "embed_tokens.weight"
    out[emb] = sel(take(emb), 1, Pt)
    if "lm_head.weight" in sd:
        out["lm_head.weight"] = sel(take("lm_head.weight"), 1, Pt)   # UNTIED real tensor
    nrm = pre_root + "norm.weight"
    out[nrm] = sel(take(nrm), 0, Pt)
    # mtp.* tensors are dropped by conversion; delete rather than permute
    for k in [k for k in out if ".mtp" in k or k.startswith("mtp.")]:
        del out[k]

    group = dims["n_heads"] // dims["n_kv"]
    full = set(dims["full_attn_layers"])
    for l in range(dims["n_layers"]):
        pre = f"{pre_root}layers.{l}."
        Pf = np.asarray(perms["ffn"][l])
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            if pre + n in sd:
                out[pre + n] = sel(take(pre + n), 0, Pt)
        out[pre + "mlp.gate_proj.weight"] = sel(take(pre + "mlp.gate_proj.weight"), 1, Pt)
        out[pre + "mlp.gate_proj.weight"] = sel(out[pre + "mlp.gate_proj.weight"], 0, Pf)
        out[pre + "mlp.up_proj.weight"] = sel(sel(take(pre + "mlp.up_proj.weight"), 1, Pt), 0, Pf)
        out[pre + "mlp.down_proj.weight"] = sel(sel(take(pre + "mlp.down_proj.weight"), 0, Pt), 1, Pf)
        if l in full:
            out[pre + "self_attn.q_proj.weight"] = sel(take(pre + "self_attn.q_proj.weight"), 1, Pt)
            out[pre + "self_attn.k_proj.weight"] = sel(take(pre + "self_attn.k_proj.weight"), 1, Pt)
            v = sel(take(pre + "self_attn.v_proj.weight"), 1, Pt)
            o = sel(take(pre + "self_attn.o_proj.weight"), 0, Pt)
            q = out[pre + "self_attn.q_proj.weight"]
            for h in range(dims["n_kv"]):
                p = np.asarray(perms["vo"][l][h])
                # v rows [h*256 + p]
                v_rows = np.arange(v.shape[0])
                v_rows[h * _HD:(h + 1) * _HD] = h * _HD + p
                v = sel(v, 0, v_rows)
                for qh in range(group * h, group * h + group):
                    # o cols [qh*256 + p]
                    o_cols = np.arange(o.shape[1])
                    o_cols[qh * _HD:(qh + 1) * _HD] = qh * _HD + p
                    o = sel(o, 1, o_cols)
                    # q GATE half rows [qh*512 + 256 + p]  (query half untouched)
                    q_rows = np.arange(q.shape[0])
                    base = qh * _QHEAD + _HD
                    q_rows[base:base + _HD] = base + p
                    q = sel(q, 0, q_rows)
            out[pre + "self_attn.v_proj.weight"] = v
            out[pre + "self_attn.o_proj.weight"] = o
            out[pre + "self_attn.q_proj.weight"] = q
        else:
            # linear-attn layer: only the P_res in-columns move (P_lav frozen identity)
            for n in ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                      "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
                if pre + n in sd:
                    out[pre + n] = sel(take(pre + n), 1, Pt)
            if pre + "linear_attn.out_proj.weight" in sd:
                out[pre + "linear_attn.out_proj.weight"] = sel(take(pre + "linear_attn.out_proj.weight"), 0, Pt)
    return out


# ---------------------------------------------------------------- save/load perms

def save_perms(perms, path, dims=None, acknowledge_unreviewed=False):
    """npz layout: res [d_model], ffn [n_layers, d_ffn], vo [n_vo_layers, n_kv, head_dim]
    with a parallel vo_layers index array (P_vo lives only on FULL-ATTN layers, so vo is a
    DICT keyed by layer -- the vo_layers array records which layers it covers)."""
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

_RES_FULL = ["attn_q", "attn_k", "attn_v"]   # full-attn in-projections (ne0 = d_model)
_RES_ALL = ["ffn_gate", "ffn_up"]            # every layer (ne0 = d_model)


def _res_space_tensors(weights, dims):
    names = ["token_embd.weight", "output.weight"]   # untied lm_head -> output.weight present
    full = set(dims["full_attn_layers"])
    for l in range(dims["n_layers"]):
        if l in full:
            names += [f"blk.{l}.{k}.weight" for k in _RES_FULL]
        names += [f"blk.{l}.{k}.weight" for k in _RES_ALL]
    return [n for n in names if n in weights]


def optimize(weights, ttypes, qws, dims, rows_sample=16384, log=print,
             acknowledge_unreviewed=False):
    """Choose P_res + P_ffn(all layers) + P_vo(full-attn layers) minimizing the container
    objective. Mirrors the qwen3 optimizer; P_lav (linear-attn v space) is frozen identity in
    v1. Returns (perms, report). weights: {gguf_name: f16 ndarray}; ttypes: {name: k-quant};
    qws: {name: imatrix means} (missing -> unweighted). vo is a DICT keyed by full-attn layer.
    """
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
    full = set(dims["full_attn_layers"])
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
        if l in full and on in weights:
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
        "note": "P_lav frozen identity (v1); P_vo full-attn layers only",
    }
    return perms, report


# ---------------------------------------------------------------- imatrix permute

def permute_imatrix(src, dst, perms, dims, acknowledge_unreviewed=False):
    """Copy an imatrix GGUF and permute every recognized .in_sum2 along ne[0] to match the
    perms. Linear-attn projection tensors whose input_perm is not recognized (P_lav frozen)
    are left unchanged -- consistent with the v1 identity freeze."""
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
    """Logits-equality oracle. Runs the reference forward, applies the permutation by writing
    the permuted state dict back (load_state_dict strict=False; re-tie if the model ties
    embeddings), and re-runs. Returns (max|dlogit|, rel). Mutates `model` in place.

    Uses the apply_perms + reload path (not an in-place mutator) so the SAME tested tensor
    edits that build the permuted checkpoint are what the gate verifies."""
    _guard(acknowledge_unreviewed)
    import torch
    model.eval()
    with torch.no_grad():
        ref = model(ids).logits.clone()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    out = apply_perms(sd, perms, dims, consume=True,
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
