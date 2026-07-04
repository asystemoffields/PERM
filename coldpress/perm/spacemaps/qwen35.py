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
    n_layers = int(config.num_hidden_layers)
    # full-attn layer indices: prefer an explicit layer_types list; else the 3,7,... pattern
    full = []
    lt = getattr(config, "layer_types", None)
    if lt:
        full = [i for i, t in enumerate(lt) if "full" in str(t).lower()]
    else:
        interval = getattr(config, "full_attention_interval", 4)
        full = [i for i in range(n_layers) if (i + 1) % interval == 0]
    return {
        "d_model": int(config.hidden_size),
        "d_ffn": int(config.intermediate_size),
        "n_layers": n_layers,
        "n_kv": int(getattr(config, "num_key_value_heads", 4)),
        "n_heads": int(getattr(config, "num_attention_heads", 16)),
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


def g3_check(model, perms, dims, ids, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    raise NotImplementedError(
        "g3_check for qwen35 needs an apply_perms_inplace path; use apply_perms + reload, "
        "and gate on rel < 1e-4 before trusting. Author the in-place path with Fable review.")
