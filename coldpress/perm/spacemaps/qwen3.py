#!/usr/bin/env python3
"""Space map: Qwen3 dense (arch qwen3; Qwen3-0.6B .. 32B dense). REFERENCE IMPLEMENTATION.

Derivation (Fable, 2026-07-03):
  res  [d_model]  embd cols + (explicit lm_head cols) + q/k/v/gate/up in-cols + o/down
                  out-rows + attn_norm/ffn_norm/output_norm gains. RMSNorm commutes with
                  permutation; tying preserved (logits = h @ embd^T, both permuted by res).
  ffn  [d_ffn] per layer   gate/up out-rows + down in-cols (SwiGLU elementwise).
  vo   [head_dim] per (layer, kv-head)   v out-rows of the kv-head + o in-cols of its GQA
                  q-heads. Position-free (attention weights come from q/k only).
FORBIDDEN: q/k head_dim (RoPE pairs (i, i+d/2)); q_norm/k_norm gains live in q/k head space.
Checkpoint quirk: explicit lm_head.weight present despite the tie flag -> permuted with
embd (G3 catches it if not). Conversion applies NO value transform for qwen3.

Everything below is pure index reordering (bit-exact moves); the G3 fp32 logits-equality
gate is the correctness oracle.
"""
import numpy as np

from ..objective import tensor_err, col_log_rms

MODEL_TYPE = "qwen3"
ARCH = "qwen3"
RES_TENSORS_PER_LAYER = ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"]


# ---------------------------------------------------------------- dims

def dims_from_config(config):
    def g(*names, default=None):
        for n in names:
            if hasattr(config, n) and getattr(config, n) is not None:
                return getattr(config, n)
        return default
    d_model = g("hidden_size")
    n_heads = g("num_attention_heads")
    n_kv = g("num_key_value_heads", default=n_heads)
    head_dim = g("head_dim", default=(d_model // n_heads if d_model and n_heads else None))
    return {
        "d_model": int(d_model),
        "d_ffn": int(g("intermediate_size")),
        "n_layers": int(g("num_hidden_layers")),
        "n_heads": int(n_heads),
        "n_kv": int(n_kv),
        "head_dim": int(head_dim),
    }


def _gqa_group(dims):
    return dims["n_heads"] // dims["n_kv"]


# ---------------------------------------------------------------- perms container

def identity_perms(dims):
    return {
        "res": np.arange(dims["d_model"]),
        "ffn": [np.arange(dims["d_ffn"]) for _ in range(dims["n_layers"])],
        "vo": [[np.arange(dims["head_dim"]) for _ in range(dims["n_kv"])]
               for _ in range(dims["n_layers"])],
    }


def save_perms(perms, path, dims=None):
    np.savez_compressed(
        path,
        res=perms["res"],
        ffn=np.stack(perms["ffn"]),
        vo=np.stack([np.stack(hs) for hs in perms["vo"]]),
    )


def load_perms(path):
    z = np.load(path)
    n_layers = z["ffn"].shape[0]
    n_kv = z["vo"].shape[1]
    return {
        "res": z["res"],
        "ffn": [z["ffn"][l] for l in range(n_layers)],
        "vo": [[z["vo"][l, h] for h in range(n_kv)] for l in range(n_layers)],
    }


def _check_perm(p, n):
    import torch
    p = np.asarray(p)
    assert p.shape == (n,) and np.array_equal(np.sort(p), np.arange(n)), \
        f"not a permutation of {n}"
    return torch.from_numpy(np.ascontiguousarray(p)).long()


# ---------------------------------------------------------------- imatrix/Hessian axis

def o_input_index(perms, layer, dims):
    """Column index for attn_output ne[0] (n_heads*head_dim): per-q-head blocks, q-head q
    uses P_vo[l][q // gqa_group]."""
    hd, nh = dims["head_dim"], dims["n_heads"]
    group = _gqa_group(dims)
    idx = np.arange(nh * hd)
    for q in range(nh):
        ph = np.asarray(perms["vo"][layer][q // group])
        idx[q * hd:(q + 1) * hd] = q * hd + ph
    return idx


def input_perm(gguf_name, perms, dims=None):
    """Permutation on this GGUF tensor's ne[0] (input) axis, or None.
    dims is inferred from perms when not given (needed only for attn_output)."""
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
        if dims is None:
            dims = _dims_from_perms(perms)
        return o_input_index(perms, layer, dims)
    return None


def _dims_from_perms(perms):
    return {
        "d_model": len(perms["res"]),
        "d_ffn": len(perms["ffn"][0]),
        "n_layers": len(perms["ffn"]),
        "n_kv": len(perms["vo"][0]),
        "head_dim": len(perms["vo"][0][0]),
        # n_heads unknown from perms alone; attn_output needs it. Recover via o-proj width
        # at apply time; input_perm for attn_output requires explicit dims from the caller.
        "n_heads": None,
    }


# ---------------------------------------------------------------- apply (state dict)

def apply_perms(sd, perms, dims, consume=False):
    """New HF state dict with permutations applied (pure index_select)."""
    import torch
    nl, nkv, hd, nh = dims["n_layers"], dims["n_kv"], dims["head_dim"], dims["n_heads"]
    group = _gqa_group(dims)
    P = _check_perm(perms["res"], dims["d_model"])

    def take(name):
        return sd.pop(name) if consume else sd[name]

    def cols(w, p):
        return w.index_select(1, p)

    def rows(w, p):
        return w.index_select(0, p)

    out = {name: w for name, w in list(sd.items())}
    out["model.embed_tokens.weight"] = cols(take("model.embed_tokens.weight"), P)
    if "lm_head.weight" in sd:
        out["lm_head.weight"] = cols(take("lm_head.weight"), P)
    out["model.norm.weight"] = take("model.norm.weight").index_select(0, P)

    for l in range(nl):
        pre = f"model.layers.{l}."
        Pf = _check_perm(perms["ffn"][l], dims["d_ffn"])
        out[pre + "input_layernorm.weight"] = take(pre + "input_layernorm.weight").index_select(0, P)
        out[pre + "post_attention_layernorm.weight"] = take(pre + "post_attention_layernorm.weight").index_select(0, P)
        out[pre + "self_attn.q_proj.weight"] = cols(take(pre + "self_attn.q_proj.weight"), P)
        out[pre + "self_attn.k_proj.weight"] = cols(take(pre + "self_attn.k_proj.weight"), P)
        v = cols(take(pre + "self_attn.v_proj.weight"), P)
        v_idx = torch.arange(nkv * hd)
        for h in range(nkv):
            ph = _check_perm(perms["vo"][l][h], hd)
            v_idx[h * hd:(h + 1) * hd] = h * hd + ph
        out[pre + "self_attn.v_proj.weight"] = rows(v, v_idx)
        o = rows(take(pre + "self_attn.o_proj.weight"), P)
        o_idx = torch.arange(nh * hd)
        for q in range(nh):
            ph = _check_perm(perms["vo"][l][q // group], hd)
            o_idx[q * hd:(q + 1) * hd] = q * hd + ph
        out[pre + "self_attn.o_proj.weight"] = cols(o, o_idx)
        out[pre + "mlp.gate_proj.weight"] = rows(cols(take(pre + "mlp.gate_proj.weight"), P), Pf)
        out[pre + "mlp.up_proj.weight"] = rows(cols(take(pre + "mlp.up_proj.weight"), P), Pf)
        out[pre + "mlp.down_proj.weight"] = cols(rows(take(pre + "mlp.down_proj.weight"), P), Pf)
    return out


def apply_perms_inplace(model, perms, dims):
    """Permute a loaded HF model's parameters in place."""
    import torch
    nl, nkv, hd, nh = dims["n_layers"], dims["n_kv"], dims["head_dim"], dims["n_heads"]
    group = _gqa_group(dims)
    with torch.no_grad():
        sd = dict(model.named_parameters())
        P = _check_perm(perms["res"], dims["d_model"])
        if hasattr(model, "lm_head") and model.lm_head is not None and \
           model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            # tied: named_parameters dedups; permuting embed_tokens covers both roles
            pass
        sd["model.embed_tokens.weight"].data = sd["model.embed_tokens.weight"].data.index_select(1, P)
        if "lm_head.weight" in sd and \
           sd["lm_head.weight"].data_ptr() != sd["model.embed_tokens.weight"].data_ptr():
            sd["lm_head.weight"].data = sd["lm_head.weight"].data.index_select(1, P)
        sd["model.norm.weight"].data = sd["model.norm.weight"].data.index_select(0, P)
        for l in range(nl):
            pre = f"model.layers.{l}."
            Pf = _check_perm(perms["ffn"][l], dims["d_ffn"])
            for n in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                sd[pre + n].data = sd[pre + n].data.index_select(0, P)
            for n in ["self_attn.q_proj.weight", "self_attn.k_proj.weight"]:
                sd[pre + n].data = sd[pre + n].data.index_select(1, P)
            v_idx = torch.arange(nkv * hd)
            for h in range(nkv):
                ph = _check_perm(perms["vo"][l][h], hd)
                v_idx[h * hd:(h + 1) * hd] = h * hd + ph
            n = pre + "self_attn.v_proj.weight"
            sd[n].data = sd[n].data.index_select(1, P).index_select(0, v_idx)
            o_idx = torch.arange(nh * hd)
            for q in range(nh):
                ph = _check_perm(perms["vo"][l][q // group], hd)
                o_idx[q * hd:(q + 1) * hd] = q * hd + ph
            n = pre + "self_attn.o_proj.weight"
            sd[n].data = sd[n].data.index_select(0, P).index_select(1, o_idx)
            for n in ["mlp.gate_proj.weight", "mlp.up_proj.weight"]:
                sd[pre + n].data = sd[pre + n].data.index_select(1, P).index_select(0, Pf)
            n = pre + "mlp.down_proj.weight"
            sd[n].data = sd[n].data.index_select(0, P).index_select(1, Pf)


# ---------------------------------------------------------------- G3 gate

def g3_check(model, perms, dims, ids):
    """Logits equality between original and permuted model. Returns (max|dlogit|, rel).
    `model` is a loaded fp32 HF model; `ids` a LongTensor [1, T]. Mutates `model` in place."""
    import torch
    model.eval()
    with torch.no_grad():
        ref = model(ids).logits.clone()
    apply_perms_inplace(model, perms, dims)
    with torch.no_grad():
        new = model(ids).logits
    d = (ref - new).abs().max().item()
    rel = d / ref.abs().max().item()
    return d, rel


# ---------------------------------------------------------------- optimize

def _res_space_tensors(weights, dims):
    names = ["token_embd.weight", "output.weight"]
    for l in range(dims["n_layers"]):
        for k in RES_TENSORS_PER_LAYER:
            names.append(f"blk.{l}.{k}.weight")
    return [n for n in names if n in weights]


def _res_objective(weights, ttypes, qws, dims, perm, rows_sample):
    total, per = 0.0, {}
    for n in _res_space_tensors(weights, dims):
        e = tensor_err(weights[n], ttypes[n], qws.get(n), perm=perm, rows_sample=rows_sample)
        per[n] = e
        total += e
    return total, per


def _build_res_keys(weights, dims):
    keys, profs = {}, []
    for n in _res_space_tensors(weights, dims):
        profs.append((n, col_log_rms(weights[n])))
    stand = [(p - p.mean()) / (p.std() + 1e-9) for _, p in profs]
    keys["mean-logrms"] = np.mean(stand, axis=0)
    emb = [p for n, p in profs if n == "token_embd.weight"]
    if emb:
        keys["embd-logrms"] = emb[0]
    blk = [s for (n, _), s in zip(profs, stand) if n.startswith("blk.")]
    if blk:
        keys["blk-logrms"] = np.mean(blk, axis=0)
    return keys


def optimize(weights, ttypes, qws, dims, rows_sample=16384, log=print):
    """Choose permutations minimizing the container objective.

    weights : {gguf_name: f32 ndarray [nrow, ne0]}  (f16 weights; mmap views ok)
    ttypes  : {gguf_name: target k-quant type string}
    qws     : {gguf_name: imatrix means [ne0]}  (missing -> ref/unweighted path)
    Returns (perms, report)."""
    import time
    t0 = time.time()
    # residual space
    base, _ = _res_objective(weights, ttypes, qws, dims, None, rows_sample)
    res_perm, res_val, res_name = np.arange(dims["d_model"]), base, "identity"
    for kname, key in _build_res_keys(weights, dims).items():
        perm = np.argsort(key, kind="stable")
        val, _ = _res_objective(weights, ttypes, qws, dims, perm, rows_sample)
        log(f"  res sort[{kname}]: {val:.6g} ({(val/base-1)*100:+.2f}%)")
        if val < res_val:
            res_perm, res_val, res_name = perm, val, kname
    log(f"  res -> {res_name} ({(res_val/base-1)*100:+.2f}%)")

    ffn_perms, ffn_val, ffn_base = [], 0.0, 0.0
    vo_perms, vo_val, vo_base = [], 0.0, 0.0
    hd, nkv, nh = dims["head_dim"], dims["n_kv"], dims["n_heads"]
    group = _gqa_group(dims)
    for l in range(dims["n_layers"]):
        # ffn_down
        fn = f"blk.{l}.ffn_down.weight"
        fb = tensor_err(weights[fn], ttypes[fn], qws.get(fn))
        fp = np.argsort(col_log_rms(weights[fn]), kind="stable")
        fv = tensor_err(weights[fn], ttypes[fn], qws.get(fn), perm=fp)
        if fv >= fb:
            fp, fv = np.arange(dims["d_ffn"]), fb
        ffn_perms.append(fp)
        ffn_val += fv
        ffn_base += fb
        # attn_output V->O per kv-head
        on = f"blk.{l}.attn_output.weight"
        x = np.asarray(weights[on])
        ob = tensor_err(x, ttypes[on], qws.get(on))
        heads = []
        for h in range(nkv):
            # kv-head h feeds q-heads {q : q//group == h}; use the first two consecutive
            # o-blocks of that kv-head to build a shared within-head key
            qs = [q for q in range(nh) if q // group == h]
            key = np.zeros(hd)
            for q in qs:
                c0 = q * hd
                key = key + col_log_rms(x[:, c0:c0 + hd])
            heads.append(np.argsort(key, kind="stable"))
        full = np.arange(nh * hd)
        for q in range(nh):
            full[q * hd:(q + 1) * hd] = q * hd + heads[q // group]
        ov = tensor_err(x, ttypes[on], qws.get(on), perm=full)
        if ov >= ob:
            heads = [np.arange(hd) for _ in range(nkv)]
            ov = ob
        vo_perms.append(heads)
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
    }
    return perms, report


# ---------------------------------------------------------------- imatrix permute

def permute_imatrix(src, dst, perms, dims):
    """Copy an imatrix GGUF and permute every .in_sum2 along ne[0] to match the perms."""
    import shutil
    from gguf import GGUFReader
    shutil.copyfile(src, dst)
    r = GGUFReader(dst, "r+")
    n_patched = 0
    for t in r.tensors:
        if not t.name.endswith(".in_sum2"):
            continue
        base = t.name[:-len(".in_sum2")]
        idx = input_perm(base, perms, dims)
        if idx is None:
            continue
        v = t.data
        assert v.ndim == 1, (t.name, v.shape)
        v[...] = v[np.asarray(idx)]
        n_patched += 1
    del r
    return n_patched
