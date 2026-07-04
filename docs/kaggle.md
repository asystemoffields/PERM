# Running coldpress measurement kernels on Kaggle

The local box does seconds-scale smoke tests only; all *measured* ppl/gate numbers come
from single-VM Kaggle GPU kernels that rebuild everything deterministically from pins. This
doc collects the Kaggle-specific operational knowledge that used to live in the research
runbook. The package CLI is arch-agnostic; a kernel just invokes the same stages with the
GPU flags.

## Kernel shape

A kernel: git-clones `llama.cpp` @ `039e20a2` and builds Release (CUDA), `snapshot_download`s
the pinned model rev, converts to f16, then runs the coldpress stages and evaluates
`llama-perplexity -c 512 -ngl 99` for every arm (stock ladder + coldpress arms) in ONE
kernel so all comparisons are within a single VM.

- CUDA build flags (known-good): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
  -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. `coldpress
  build-llamacpp --cuda --cuda-arch 75` emits this set. NO pipes on build commands — pipe
  exit codes mask failures; redirect to log files and assert binaries exist after.
- Accelerator: ALWAYS `--accelerator NvidiaTeslaT4` (P100 default gives `no kernel image`
  at the first CUDA op).
- Hessians/teacher for models >8B: shard Hessians (`--hessian-shards 2..4`) and run the
  teacher in bf16 on 2×T4 (`device_map="auto"`) or 8-bit.

## Operating point (mechanical)

target(P) = 0.95 × min{stock wiki ppl at bytes ≤ bytes(P)×1.001}; pick the k-quant P with
the smaller needed reduction; tie-break (≤3 points) toward lower bpw. Record the decision in
the ledger BEFORE measuring any coldpress arm at that point.

## Kaggle landmine table (each cost a debugging cycle)

| symptom | cause | fix |
|---|---|---|
| push "successful" but kernel missing | 5-slot silent drop | poll `kernels status` ≤90s after push; fresh suffix per retry, NEVER reuse a slug |
| GPU kernel `no kernel image` at first op | P100 default | ALWAYS `--accelerator NvidiaTeslaT4` |
| dataset not found in kernel | mount-path drift / dataset race | glob `/kaggle/input/**/<file>` recursive; fail fast; wait for `datasets status ready` before push |
| CPU kernel cancelled ~9–12h | session cap | GPU kernels for evals (~3 min/eval vs ~2h) |
| local box OOM/thermal | >1 heavy process | Kaggle for everything but seconds-scale smokes |

## Ship

Ship the artifact + `paper/METHOD.md` summary + ablation table + the reproduce pointer to
this repo. State the bytes-equal claim and the measured deltas with CIs verbatim — no
rounding up.
