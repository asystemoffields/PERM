# coldpress

**Container-aware encoding of standard GGUF k-quants.** At a fixed low-bit operating
point, `coldpress` produces a GGUF whose held-out perplexity is lower than the best
stock-`llama.cpp` artifact of equal-or-smaller file size — while staying a **byte-map
-identical standard GGUF** that loads in stock `llama.cpp` with no code changes. Only the
*encoder that chooses the bits* differs.

```
pip install coldpress          # or: git clone … && pip install -e .
coldpress quantize Qwen/Qwen3-0.6B --preset Q2_K --out qwen3-0.6b-coldpress-Q2_K.gguf
```

One command; sensible defaults; every stage resumable from a `--workdir` cache; every stage
guarded by built-in verification gates. CUDA used if present, CPU works. No Kaggle
assumptions, no absolute paths.

## What it does

Three stages, each independently ablatable, all preserving the standard container:

| stage | what | tier |
|---|---|---|
| **PERM** | function-preserving channel permutation that regroups k-quant sub-blocks so channels sharing a scale have similar magnitude. Zero bytes, zero runtime, exactly reversible (pure index reordering; the imatrix is permuted to match, free). | 2 (needs a spacemap) |
| **EF** | GPTQ-style Hessian error feedback encoded directly into the standard two-level k-quant container; `act_order` always on. Also gives `token_embd`/`output` the imatrix treatment stock `llama.cpp` skips. | 1 (any arch) |
| **NORM / E3B** | distill the container's free F32 norm gains (NORM) and fp16 superblock scales (E3B) against the FP teacher (KL on calibration text). | 1 (any arch) |

## Two tiers

- **Tier 1 (universal)** — EF + NORM/E3B work on **any architecture `llama.cpp` supports**.
  They need only the GGUF (types/shapes/names), Hessians keyed by GGUF tensor name (the
  generic collector maps every `nn.Linear` to its GGUF tensor via gguf's `TensorNameMap`,
  with a shape cross-check), and the FP teacher.
- **Tier 2 (spacemap archs)** — adds PERM, via a registry keyed by `config.model_type`.
  Ships with **qwen3** (implemented + gated); **qwen35** and **gemma4** are Fable
  derivations pending review (raise unless `--acknowledge-unreviewed`). An unknown arch
  falls back to tier 1 with a friendly message pointing at the spacemap authoring guide.

## Results (Qwen3-0.6B; wikitext-2-test primary / fineweb-heldout no-Goodhart)

Operating point: the Q2_K preset map (347.3 MB; 113×Q2_K + 84×Q3_K + Q6_K output), chosen
by a rule locked before any coldpress measurement. Baseline to beat: the best stock
artifact at ≤ equal bytes = IQ3_XXS (345.9 MB), wiki 43.4707 / fineweb 36.9911. All numbers
from single-VM Kaggle T4 kernels (`llama.cpp` @ `039e20a2`, `-c 512`).

| arm (Q2_K map unless noted) | wiki ppl | fineweb ppl |
|---|---|---|
| f16 reference | 21.4689 | 21.1220 |
| stock Q2_K (imatrix) | 44.4069 | 38.9027 |
| stock IQ3_XXS (imatrix; the ≤-bytes baseline) | 43.4707 | 36.9911 |
| E1 = PERM only (stock encoder) | 40.5700 | 38.2178 |
| **E2 = EF only (our encoder, stock channel order)** | **36.8667** | **33.0054** |
| E1+E2 naive composition (storage-order GPTQ) | 67.0726 | 52.2424 |

**Gates (paired bootstrap, 583 wiki / 494 fineweb chunks, 10k resamples):**
- **G1 PASS**: EF vs IQ3_XXS wiki **−15.19%** [95% CI −16.11, −14.26] (prereg bar −5%).
- **G2 PASS**: fineweb **−10.78%** [−11.70, −9.85] (bar −2.5%).
- PERM alone also clears G1: **−6.67%** [−7.66, −5.68] from a pure reorder through the stock
  encoder (function-preserving to 2.7e-6 rel logits).

Three measured failure modes are baked into the tool as hard rules: `act_order` is not a
user knob (naive PERM×EF is catastrophic, +51% wiki), distillation students are constructed
untied, and no subprocess step is pipe-masked. See `paper/METHOD.md` for the full method,
negatives, and citations.

## CLI

```
coldpress onboard <hf_id>     # config fetch, divisibility check, arch/tier report, plan
coldpress calibrate <hf_id>   # download; build/locate llama.cpp; f16; imatrix; Hessians; teacher
coldpress perm <hf_id>        # (tier 2) optimize perms; G3 gate; permute imatrix; f16
coldpress encode <hf_id>      # stock-quantize; EF; byte-parity gate
coldpress distill <hf_id>     # --norm (fast) and/or --scales (full E3B); untied student
coldpress verify --gguf F     # typemap gate; smoke gen; optional --ppl-check corpus.txt
coldpress quantize <hf_id>    # the whole chain
coldpress build-llamacpp --dest DIR [--cuda]   # clone the pinned commit and build
```

`llama.cpp` is located via `--llama-cpp DIR`, `$COLDPRESS_LLAMACPP`, or `PATH`; or built
from the pinned commit `039e20a2` with `coldpress build-llamacpp`. See `RECIPE.md` for the
per-model runbook and `docs/kaggle.md` for running the measurement kernels on shared GPUs.

## Adding an architecture (authoring a spacemap)

PERM needs a **spacemap** — a declaration of every permutable internal space and the exact,
value-free tensor edits that realize a coordinated permutation, keyed by `config.model_type`
under `coldpress/perm/spacemaps/`. `qwen3.py` is the reference. The correctness oracle is
the **G3 gate**: fp32 logits equality between the original and permuted model (rel < 1e-4,
hard fail) on random permutations. See `docs/spacemaps.md` and `RECIPE.md §10`.

## License

See `LICENSE` (TODO: Alex chooses — suggest MIT or Apache-2.0).
