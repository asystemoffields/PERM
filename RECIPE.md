# coldpress RECIPE — per-model runbook

The full pipeline is executable **without Fable-level reasoning**: every stage has a command
and a VERIFICATION GATE with an exact pass criterion. You do not need insight to avoid
mistakes — the gates catch them. If a gate fails and the landmine table (§9) doesn't cover
it, STOP and flag Alex rather than improvising.

The one judgment-heavy step — deriving permutation spaces for a NEW architecture — is NOT
part of this runbook. Use a spacemap from `coldpress/perm/spacemaps/` (§4, `docs/spacemaps.md`).
If none exists for your target architecture: STOP, flag Alex.

Everything below assumes the package CLI. To run the *measurement* kernels on shared GPUs
(Kaggle), see `docs/kaggle.md`; the local box does seconds-scale smokes only.

## 0. What you produce

For a target model M and k-quant preset P (usually Q2_K or Q3_K_S): a standard GGUF with the
SAME per-tensor type map and ≤ byte count as stock `llama-quantize` output, but 10–17% lower
held-out ppl (small models, ≤3bpw; expect less at larger scale). Stages, each independently
ablatable: PERM (free reorder) → EF (Hessian error feedback, `act_order` ALWAYS) → NORM/E3B.

## 1. Onboard (STOP conditions)

```
coldpress onboard <hf_id> --revision <pin a commit hash>
```
Checks — ALL must pass or STOP:
- [ ] every matmul ne[0] (hidden, ffn, n_heads·head_dim, MoE expert dims) divisible by 256
      (the tool prints this; a failure is a hard STOP — k-quant containers need 256-multiples)
- [ ] architecture supported by `llama.cpp` @ `039e20a2`
- [ ] a spacemap exists for this arch (else tier 1: EF + NORM only, PERM skipped) — the tool
      reports the tier
- [ ] `convert_hf_to_gguf.py` applies NO value permutation for this arch (llama-family
      permutes q/k; qwen/gemma do not; if it permutes, the spacemap must account for it and
      only Fable signs that off)

## 2. Calibrate (build inputs)

```
coldpress calibrate <hf_id> --workdir W [--llama-cpp DIR | after: coldpress build-llamacpp]
                            [--calib FILE] [--hessian-shards N] [--cuda]
```
Downloads the model rev; builds/locates `llama.cpp`; converts to f16; runs `llama-imatrix`
on the calibration text (default: bundled fineweb-edu slice); collects Hessians (generic
collector, keyed by GGUF tensor name, shape-cross-checked); caches the teacher top-256
logits. All artifacts into `W/`, each manifest-guarded (resumable).
GATE: Hessians are PSD and shape-match the GGUF; teacher chunks match the `ids/top_i/top_lp/tail`
schema. For models >8B use `--hessian-shards 2..4`.

## 3. PERM (tier 2 only; `act_order` handles the rest)

```
coldpress perm <hf_id> --workdir W [--acknowledge-unreviewed for qwen35/gemma4]
```
Optimizes permutations against the exact container objective at the target preset map,
applies them to a copy of the checkpoint, and converts to f16 + permutes the imatrix.
GATE **G3** (hard): fp32 logits equality between original and permuted model, rel < 1e-4
(expect ~1e-6). A failure means the permutation is not function-preserving — STOP.
GATE: permuted artifact's per-tensor type map identical to stock; file size ≤ stock (a small
`general.name` metadata delta is normal).
TRIPWIRE (prereg discipline): measure the PERM-only ppl at P before composing. If PERM is
worse than stock at P, ship EF-only and flag Alex.

## 4. EF (`act_order` is NOT optional)

```
coldpress encode <hf_id> --workdir W --preset P
```
Stock-quantizes the (permuted) f16 at P, then EF-encodes with `act_order=True` (immutable
default; exposed only as the hidden `--unsafe-storage-order` for ablation — NEVER ship it).
GATE **byte-parity** (hard): per-tensor type+size == stock map, total ≤ stock.
EXPECT: EF beats stock by >5% wiki at the operating point (measure in a kernel; if it does
not, check `.efstats.json` + the landmine table).

## 5. NORM / E3B distill

```
coldpress distill <hf_id> --workdir W --norm [--scales]
```
`--norm` tunes the F32 norm gains (fast); `--scales` additionally tunes every superblock
d/dmin (full E3B, `--device cuda` for speed). The student is constructed UNTIED
(`tie_word_embeddings=False`) — mandatory, because the GGUF stores embd and output at
different quant types and a tied student silently clobbers one with the other.
GATE: val teacher-KL decreases ≥5% (reference: −21% NORM-only on stock Q2_K at 0.6B).

## 6. Final gates, smoke, ship

```
coldpress verify --gguf FINAL --stock STOCK --ppl-check corpus.txt
```
- byte-parity gate again on the final artifact.
- Smoke: `llama-cli` loads and answers a spot prompt coherently.
- Optional A/B ppl vs stock (`parse_ppl` paired bootstrap).
- G1/G2 (measured in a kernel): best artifact wiki ≤ 0.95·best-stock-≤-bytes and fineweb ≤
  0.975·same, both with paired-bootstrap CI clear of the bar.
- Ship to HF as `Asystemoffields/<Model>-COLDPRESS-<P>-GGUF` with the method summary
  (`paper/METHOD.md`), the ablation table, and the measured deltas + CIs verbatim (no
  rounding up). State the bytes-equal claim.

## 9. Landmine table (general; Kaggle-specific ones live in `docs/kaggle.md`)

| symptom | cause | fix |
|---|---|---|
| CUDA cmake fails in seconds | driver stub link | the known-good flag set incl. `GGML_CUDA_NO_VMM=ON` (built in to `build-llamacpp --cuda`); never pipe build output |
| `llama-quantize` exit 127 after "successful" build | a pipe masked the build failure | the tool redirects to log files and asserts binaries exist — never pipe-mask |
| permuted model breaks logits | explicit `lm_head.weight` in checkpoint not permuted | spacemaps handle it; the G3 gate catches it |
| PERM+EF artifact catastrophically bad | GPTQ dumps error into last (= most important, post-sort) columns | `act_order=True` (default); do not "simplify" it away |
| our encoder ~5% worse than llama-quantize | any deviation from ggml's exact scale search | run the encoder-faithfulness gate: median byte-diff < 0.1% on real weights |
| embd/output look unweighted in stock | imatrix collector only tracks blk.* | expected; EF supplies the lm-head Hessian + embd column weights |
| student distill loads wrong embd | tied student + untied GGUF | `tie_word_embeddings=False` (baked in) |
| diag-weighted err says EF is worse | GPTQ trades diagonal err for correlated compensation | ignore that metric for EF; only real ppl decides |

## 10. What still requires Fable

- Spacemap derivation for an architecture not in `coldpress/perm/spacemaps/` (the
  RoPE/tying/norm-structure invariance analysis). Available: qwen3 (done). qwen35 + gemma4:
  Fable derivations pending review — gated behind `--acknowledge-unreviewed`; do NOT ship
  their output until Fable signs off and G3 passes on the real checkpoint.
- Changing gates, claim shape, or the decision rule.
- Interpreting a result that contradicts this runbook's expectations.
