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

## Results (Qwen3-0.6B, Q2_K operating point, 347MB — identical bytes to stock)

| arm | WikiText-2 ppl | FineWeb-held ppl |
|---|---|---|
| stock `llama-quantize` Q2_K | 44.41 | 38.90 |
| best stock artifact ≤ bytes (IQ3_XXS) | 43.47 | 36.99 |
| PERM only (stock encoder, pure reorder) | 40.57 | 38.22 |
| EF only | 33.97 | 30.47 |
| PERM + EF | 33.67 | 30.12 |
| **PERM + EF + NORM (the recipe)** | **30.04** | **27.01** |
| f16 reference | 21.47 | 21.12 |

**−30.9% [95% CI −31.6, −30.1] vs the best stock artifact of equal-or-smaller size**
(−32.3% vs stock at identical bytes); the 347MB artifact matches stock Q3_K_S (390MB)
within noise. All arms measured within a single kernel run, same binary, same VM.


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

Licensed under the [MIT License](LICENSE).
