#!/usr/bin/env python3
# =============================================================================
#  ##  PENDING FABLE REVIEW + G3 GATE ON REAL WEIGHTS  ##
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
    return None  # attn_output ne0 perm on sliding layers: author with the GQA composition


def _prefix(sd):
    """Gemma multimodal checkpoints nest the text stack under model.language_model.*."""
    if any(k.startswith("model.language_model.") for k in sd):
        return "model.language_model."
    return "model."


def apply_perms(sd, perms, dims, consume=False, acknowledge_unreviewed=False):
    """Apply P_res + P_ffn + P_vo(sliding) to a Gemma4 HF state dict (text stack)."""
    _guard(acknowledge_unreviewed)
    import torch
    P = np.asarray(perms["res"])
    pre_root = _prefix(sd)

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


def g3_check(model, perms, dims, ids, acknowledge_unreviewed=False):
    _guard(acknowledge_unreviewed)
    raise NotImplementedError(
        "g3_check for gemma4 needs an apply_perms_inplace path (and, for -it/vision, a "
        "multimodal forward); author with Fable review and gate on rel < 1e-4.")
