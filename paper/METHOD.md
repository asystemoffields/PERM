# COLDPRESS: container-aware encoding of standard GGUF k-quants

**Claim (matched-condition):** at a fixed low-bit operating point, a GGUF whose held-out
perplexity is ≥5% lower than the best stock-llama.cpp artifact of equal-or-smaller file
size, for the same model, same eval binary, same calibration text. The file is a standard
GGUF: identical per-tensor type map to the stock artifact, byte-identical layout, loads in
stock llama.cpp with no code changes. Only the *encoder that chooses the bits* differs.

## Model / conditions

- Qwen3-0.6B (rev c1899de), llama.cpp @ 039e20a2 for all quantization+eval.
- Primary eval: wikitext-2-raw test, `llama-perplexity -c 512`. Secondary (no-Goodhart):
  fineweb-edu held-out slice, disjoint from calibration.
- Calibration (identical for every arm, stock and COLDPRESS): fineweb-edu slice, ~3MB.
- Baselines: all stock llama-quantize presets (K + IQ families) with imatrix from the
  same calibration text. TODO ladder table.

## The three stages

### PERM — container-aware channel permutation (the novel core)

k-quant containers share quantization scales across *consecutive runs of 16 or 32
channels along ne[0]*, nested in 256-wide superblocks whose fp16 scale quantizes the
sub-scales to 4-6 bits. Which channels land in a block together is an artifact of
arbitrary channel order — but the function of a transformer is invariant under
coordinated permutations of its internal spaces. PERM permutes:

- the **residual stream** (1024): token_embd + output columns, q/k/v/gate/up input
  columns, o/down output rows, all residual-space norm gains — preserves embedding tying;
- each layer's **ffn hidden space** (3072): gate/up output rows + down input columns;
- each (layer, kv-head)'s **V→O head_dim space** (128): v output rows + the o input
  columns of its two GQA query heads. (RoPE forbids q/k head_dim permutation; the
  V→O path is position-free.)

so that channels sharing a sub-block have similar scale. Zero byte cost, zero runtime
cost, exactly function-preserving (pure reordering; no arithmetic on weights).
Permutations chosen against the true container objective: imatrix-weighted squared
reconstruction error of a faithful reimplementation of llama.cpp's own weighted
encoder, at the tensor's exact target type from the preset map.

Nearest prior art (must-cite): PermuQuant (arXiv 2605.09503; diffusion models, NVFP4
scale groups) and CHAMP-Q (2026; W4A8, ~128-wide groups) establish the mechanism family
in 2026; RPTQ (activation-side runtime reorder) and GPTQ act_order are older cousins.
Confirmed absent in prior art: any weight-side permutation for the llama.cpp/GGUF
ecosystem; the k-quant two-level 16/32-sub-block instantiation; tied-embedding
whole-model treatment; permutation of the imatrix itself to match (exact, free).

### EF — Hessian error feedback into the container

GPTQ-style cross-column error propagation (activation second moments from the same
calibration text) encoding directly into the standard two-level k-quant containers:
container scales committed from a stock-style weighted fit, integer codes chosen by
error-feedback sweep given those scales. Nearest art (must-cite): IST-DASLab
gptq-gguf-toolkit, Intel AutoRound GGUF export, ikawrakow's imatrix machinery itself.
New here: the joint composition with PERM (feedback sweeps run in permuted order) and
the committed-two-level-grid formulation for 16-wide k-quant sub-blocks.

Additionally, EF upgrades the two largest tensors in the file: stock llama.cpp gives
token_embd and output **no imatrix at all** (the collector only tracks blk.* matmuls),
quantizing ~52% of parameters blind; COLDPRESS encodes output.weight with a proper
lm-head Hessian and token_embd with informed column weights — same container, same bytes.

### NORM — norm-gain distillation

The GGUF stores every norm gain in F32 (1-D tensor rule): ~65K free continuous
parameters in a "fully quantized" file. Post-quant, we distill them against the FP
teacher (KL on calibration text) to absorb systematic quantization bias. This is the
GGUF instantiation of Norm Tweaking (Li et al., AAAI 2024) — must-cite.

## Results (Qwen3-0.6B, wikitext-2-test primary / fineweb-heldout no-Goodhart)

Operating point: the Q2_K preset map (347.3MB; 113×Q2_K + 84×Q3_K + Q6_K output),
chosen by a rule locked before any COLDPRESS measurement. Baseline to beat: the best
stock artifact at ≤ equal bytes = IQ3_XXS (345.9MB), wiki 43.4707 / fineweb 36.9911.
All numbers below from single-VM Kaggle T4 kernels (llama.cpp @ 039e20a2, -c 512).

| arm (Q2_K map unless noted) | wiki ppl | fineweb ppl |
|---|---|---|
| f16 reference | 21.4689 | 21.1220 |
| stock Q2_K (imatrix) | 44.4069 | 38.9027 |
| stock IQ3_XXS (imatrix; the ≤-bytes baseline) | 43.4707 | 36.9911 |
| E1 = PERM only (stock encoder) | 40.5700 | 38.2178 |
| **E2 = EF only (our encoder, stock channel order)** | **36.8667** | **33.0054** |
| E1+E2 naive composition (storage-order GPTQ) | 67.0726 | 52.2424 |
| E2+E3 (+ NORM / + full scale distillation) | TBD | TBD |

**Gates (paired bootstrap, 583 (wiki) / 494 (fineweb) chunks, 10k resamples):**
- **G1 PASS**: EF vs IQ3_XXS wiki **−15.19%** [95% CI −16.11, −14.26] (prereg bar −5%).
- **G2 PASS**: fineweb **−10.78%** [−11.70, −9.85] (bar −2.5%).
- PERM alone also clears G1: −6.67% [−7.66, −5.68] from a pure reorder through the
  stock encoder (zero bytes, zero runtime, function-preserving to 2.7e-6 rel logits).
- Ship smoke: our artifact answers a spot prompt coherently at 83 t/s in stock
  llama-cli; the byte-identical-map stock Q2_K degenerates into a repetition loop.

**Negative results (kept, they matter):**
1. The IQ codebook family collapses on this 0.6B model: IQ3_XXS (3.06bpw) ≈ Q2_K
   (2.63bpw) on wiki, IQ2_M far worse than Q2_K. k-quants dominate the small-model
   low-bit regime here — which is exactly the regime our containers target.
2. Naive PERM×EF composition is catastrophic, and the diagonal-weighted error metric
   cannot see it (EF raises diag-weighted error +52% median while cutting real ppl
   17%): magnitude-sorted storage order puts the most important channels last, where
   GPTQ's sequential sweep dumps accumulated error. Fix: act_order processing decoupled
   from storage order (the committed-grid design makes sweep order free).
3. Perms optimized against the Q2_K containers are ≈neutral at Q3_K_S — the grouping
   win is container-specific, as the mechanism predicts.

Container-objective interim (PERM, pre-ppl): −9.7% residual / −30.1% ffn (−54% layer
27) / −4.8% V→O imatrix-weighted quant error; encoder faithfulness vs llama-quantize
bytes: 0.028% median byte-diff across 198 tensors.

## Reproduce

All measured numbers come from self-contained Kaggle kernels (CPU for statistics collection, GPU for all gate measurements) that rebuild
everything deterministically from pins (kaggle/evalA*, evalB, e2, e3probe in this
repo): git-clone llama.cpp @ 039e20a2 and build Release; snapshot_download
Qwen/Qwen3-0.6B @ c1899de; convert_hf_to_gguf → f16; PERM = pure reorder from
work/perms-e1.npz via src/perm_spaces.py (G3: fp32 logits equality, rel ≤ 3e-6);
imatrix permuted entry-wise by the same perms (src/perm_imatrix.py); stock
llama-quantize builds both stock and PERM arms; EF/NORM arms are encoded by
src/e2_ef.py / src/norm_distill.py and re-emitted through src/ggufio.py (G0:
re-emission is byte-identical on tensor data + KV). Eval: llama-perplexity -c 512
-t 4, full file; every gate comparison computed within a single kernel run.
Calibration inputs (fineweb-calib), eval corpora, imatrices, perms, and all src
modules ship in the kaggle dataset asystemoffields/coldpress-corpus; per-tensor
activation Hessians (X^T X over 131,072 calib tokens) and the cached teacher top-256
log-probs come from kernel coldpress-hess3.

## Citations

PermuQuant 2605.09503; CHAMP-Q; RPTQ 2304.01089; GPTQ 2210.17323; AutoRound;
IST-DASLab/gptq-gguf-toolkit; Norm Tweaking AAAI'24 (2309.02784); QuaRot 2404.00456
(rejected at design time: breaks embedding tying on tied-embd models); llama.cpp imatrix
(ikawrakow); SpinQuant 2405.16406 (rotation family context).
