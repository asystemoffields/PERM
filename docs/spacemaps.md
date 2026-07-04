# Authoring a spacemap (adding PERM support for an architecture)

PERM is tier-2: it needs a **spacemap** for the target architecture. A spacemap declares
every permutable internal space and the exact, **value-free** tensor edits that realize a
coordinated permutation of that space, keyed by `config.model_type` under
`coldpress/perm/spacemaps/`. `qwen3.py` is the reference implementation — read it first.

> The invariance analysis (which spaces are permutable given RoPE / embedding tying / norm
> structure / GQA / gating) is the one judgment-heavy step and **requires Fable**. Do not
> derive a spacemap from scratch. This doc is the *implementation* contract once a derivation
> exists.

## The interface each spacemap module exposes

```
MODEL_TYPE, ARCH                       # config.model_type and GGUF arch strings
dims_from_config(config) -> dict       # d_model, d_ffn, n_layers, n_heads, n_kv, head_dim
identity_perms(dims)                   # the identity element
save_perms(perms, path) / load_perms(path)
apply_perms(state_dict, perms, dims, consume=False)   # pure index_select on an HF sd
apply_perms_inplace(model, perms, dims)               # in-place on a loaded HF model
input_perm(gguf_tensor_name, perms, dims) -> perm|None  # ne[0]-axis perm (imatrix/Hessian)
optimize(weights, ttypes, qws, dims, rows_sample) -> (perms, report)
g3_check(model, perms, dims, ids) -> (max|dlogit|, rel)   # the correctness oracle
permute_imatrix(src, dst, perms, dims)                    # permute an imatrix GGUF to match
```

## The three canonical spaces (qwen3 family)

- **res** `[d_model]` — the residual stream: embd cols, (explicit lm_head cols if present),
  q/k/v/gate/up input cols, o/down output rows, and every residual-space norm gain. RMSNorm
  commutes with permutation; embedding tying is preserved (logits = h·embdᵀ, both permuted
  by the same `res`).
- **ffn** `[d_ffn]` per layer — gate/up output rows + down input cols (SwiGLU is elementwise).
- **vo** `[head_dim]` per (layer, kv-head) — v output rows of the kv-head + o input cols of
  its GQA query heads. Position-free (attention weights come from q/k only).

**Forbidden**: q/k head_dim (RoPE pairs dims (i, i+d/2) with fixed frequencies); q_norm/k_norm
gains live in that head space and must stay put.

## The correctness oracle: G3

The only thing that certifies a spacemap is **fp32 logits equality**: build the original
model, snapshot logits, `apply_perms_inplace`, snapshot again — `rel = max|Δlogit| /
max|logit|` must be `< 1e-4` (expect ~1e-6) on **random** permutations (identity must be
exactly 0). If G3 fails, the permutation is not function-preserving; find the tensor you
mis-edited. Run this on a TINY random model of the same config first (seconds), then on the
real checkpoint before shipping.

## Registering

Add the module under `coldpress/perm/spacemaps/` and map `config.model_type` to it in
`coldpress/perm/registry.py`. Unknown model types fall back to tier 1 (EF + NORM only).

## Unreviewed maps

`qwen35.py` and `gemma4.py` are Fable *derivations implemented but not yet reviewed*, with no
G3 gate run on real weights. They carry a loud banner and raise `NotImplementedError` unless
called with `acknowledge_unreviewed=True` (CLI: `--acknowledge-unreviewed`). Do NOT ship an
artifact built with an unreviewed map until Fable signs off AND G3 passes on the real
checkpoint.
