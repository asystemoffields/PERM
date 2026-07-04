#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# parse_ppl.py -- parse llama.cpp llama-perplexity stdout logs, compute stats.
#
# FORMAT FACTS, verified against the pinned source
#   llama.cpp tools/perplexity/perplexity.cpp  (commit 039e20a2)
#
# 1. Per-chunk running-estimate stream (default --ppl-stride unset, i.e. the
#    main perplexity() path, params.ppl_output_type == 0), line 633:
#        LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
#    perplexity_v2() (used only when --ppl-stride > 0) prints the identical
#    format at line 434. The value is the CUMULATIVE running perplexity
#    exp(total_nll_so_far / total_count_so_far), NOT a per-chunk value.
#    The stream has no newlines between entries, e.g.:
#        [1]4.3021,[2]4.5155,[3]4.9700,...
#
# 2. Final line (perplexity() only; perplexity_v2 never prints it), line 654:
#        LOG_INF("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
#    i.e. PPL with 4 decimals, +/- error with 5 decimals.
#
# 3. Tokens scored per chunk (perplexity(), lines 542 and 629):
#        const int first = n_ctx/2;            // only last half is scored
#        count += n_ctx - first - 1;           // last token has no target
#    => with -c 512: 512 - 512/2 - 1 = 255 predicted tokens per chunk.
#    n_ctx is recoverable from the header line (line 517):
#        "%s: calculating perplexity over %d chunks, n_ctx=%d, batch_size=%d,
#         n_seq=%d"
#
# RECONSTRUCTION
#    Every chunk scores the same number m of tokens, so the printed running
#    value satisfies  ppl_i = exp( (m * sum_{j<=i} nll_j) / (i*m) )
#                           = exp( mean(nll_1..nll_i) ),
#    where nll_j is the per-token mean negative log-likelihood of chunk j.
#    Hence  nll_i = i*ln(ppl_i) - (i-1)*ln(ppl_{i-1})   (with ppl_0 := 1),
#    independent of m.
#
# PRECISION NOTE
#    The running values are printed with only 4 decimals, so ln(ppl_i) carries
#    an absolute error of up to 5e-5/ppl_i, and the reconstruction multiplies
#    it by the chunk index: |err(nll_i)| <= (2i-1) * 5e-5 / min(ppl). For
#    ppl ~ 5-20 and a few hundred chunks this is O(1e-3..1e-2) per chunk --
#    small against the O(0.1-1) chunk-to-chunk nll spread, so it is fine for
#    a chunk-level paired bootstrap, but do not treat single reconstructed
#    chunk values (especially late ones) as exact.
# ---------------------------------------------------------------------------
"""Parse llama-perplexity logs: summary table and paired bootstrap deltas.

Usage:
    parse_ppl.py table <logdir-glob-or-paths...>
    parse_ppl.py delta <log_a> <log_b>
"""

import glob
import os
import re
import sys

import numpy as np

# "[%d]%.4lf," -> e.g. "[17]4.9700,"
CHUNK_RE = re.compile(r"\[(\d+)\](\d+\.\d{4}),")
# "Final estimate: PPL = %.4lf +/- %.5lf"
FINAL_RE = re.compile(r"Final estimate: PPL = (\d+\.\d{4}) \+/- (\d+\.\d{5})")
# "calculating perplexity over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d"
HEADER_RE = re.compile(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)")


def _chunk_run(text):
    """Extract the (index, running_ppl) stream, tolerant of stray bracketed
    numbers elsewhere in the log: keep the longest consecutive run of
    indices that starts at 1."""
    pairs = [(int(i), float(v)) for i, v in CHUNK_RE.findall(text)]
    runs = []
    cur = []
    for j, v in pairs:
        if cur and j == cur[-1][0] + 1:
            cur.append((j, v))
        else:
            cur = [(j, v)]
            runs.append(cur)
    starts_at_1 = [r for r in runs if r[0][0] == 1]
    if not starts_at_1:
        raise ValueError("no per-chunk '[i]ppl,' stream found")
    return max(starts_at_1, key=len)


def parse_log(path):
    """Parse one llama-perplexity stdout log.

    Returns a dict with:
        final_ppl, final_err     -- from the 'Final estimate' line (None if absent)
        running_ppl              -- np.array of printed cumulative running PPLs
        chunk_nll                -- np.array of reconstructed per-chunk mean NLLs
        n_chunks                 -- number of per-chunk entries found
        n_ctx, tokens_per_chunk  -- from the header line (None if absent);
                                    tokens_per_chunk = n_ctx - n_ctx//2 - 1
        header_n_chunks          -- planned chunk count from the header
    """
    with open(path, "r", errors="replace") as f:
        text = f.read()

    run = _chunk_run(text)
    running = np.array([v for _, v in run], dtype=np.float64)
    n = len(running)

    # nll_i = i*ln(ppl_i) - (i-1)*ln(ppl_{i-1})  (see header comment)
    i = np.arange(1, n + 1, dtype=np.float64)
    cum = i * np.log(running)                       # cumulative mean-nll * i
    chunk_nll = np.diff(cum, prepend=0.0)

    m = FINAL_RE.search(text)
    final_ppl = float(m.group(1)) if m else None
    final_err = float(m.group(2)) if m else None

    h = HEADER_RE.search(text)
    header_n_chunks = int(h.group(1)) if h else None
    n_ctx = int(h.group(2)) if h else None
    tokens_per_chunk = (n_ctx - n_ctx // 2 - 1) if n_ctx is not None else None

    return {
        "final_ppl": final_ppl,
        "final_err": final_err,
        "running_ppl": running,
        "chunk_nll": chunk_nll,
        "n_chunks": n,
        "n_ctx": n_ctx,
        "tokens_per_chunk": tokens_per_chunk,
        "header_n_chunks": header_n_chunks,
    }


def _artifact_name(path):
    name = os.path.basename(path)
    if name.endswith(".log"):
        name = name[: -len(".log")]
    if name.startswith("ppl-"):
        name = name[len("ppl-"):]
    return name


def summarize(paths):
    """Print a markdown table: artifact | final ppl | +/- err | n_chunks."""
    rows = []
    for p in paths:
        d = parse_log(p)
        rows.append((
            _artifact_name(p),
            "n/a" if d["final_ppl"] is None else f"{d['final_ppl']:.4f}",
            "n/a" if d["final_err"] is None else f"{d['final_err']:.5f}",
            str(d["n_chunks"]),
        ))
    headers = ("artifact", "final ppl", "+/- err", "n_chunks")
    widths = [max(len(h), *(len(r[k]) for r in rows)) if rows else len(h)
              for k, h in enumerate(headers)]
    def fmt(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    print(fmt(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(fmt(r))


def paired_delta(log_a, log_b, n_boot=10000, seed=0):
    """Paired bootstrap (over chunks, paired by index -- same eval text) of the
    relative perplexity delta of B vs A:

        delta = ppl_b / ppl_a - 1 = exp(mean(nll_b) - mean(nll_a)) - 1

    Returns dict with point estimate in % and a 95% percentile CI in %.
    """
    da, db = parse_log(log_a), parse_log(log_b)
    assert da["n_chunks"] == db["n_chunks"], (
        f"chunk-count mismatch: {log_a} has {da['n_chunks']}, "
        f"{log_b} has {db['n_chunks']} (logs must score the same corpus)")
    a, b = da["chunk_nll"], db["chunk_nll"]
    n = len(a)

    point_pct = (np.exp(b.mean() - a.mean()) - 1.0) * 100.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = (np.exp(b[idx].mean(axis=1) - a[idx].mean(axis=1)) - 1.0) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "point_pct": float(point_pct),
        "ci95_lo_pct": float(lo),
        "ci95_hi_pct": float(hi),
        "n_chunks": n,
        "n_boot": n_boot,
    }


def main(argv):
    if len(argv) < 2 or argv[1] not in ("table", "delta"):
        print(__doc__.strip(), file=sys.stderr)
        return 2

    if argv[1] == "table":
        paths = []
        for arg in argv[2:]:
            hits = sorted(glob.glob(arg))
            if hits:
                paths.extend(hits)
            elif os.path.exists(arg):
                paths.append(arg)
            else:
                print(f"table: no logs matched {arg!r}", file=sys.stderr)
        if not paths:
            print("table: no logs matched", file=sys.stderr)
            return 1
        summarize(paths)
        return 0

    # delta
    if len(argv) != 4:
        print("usage: parse_ppl.py delta <log_a> <log_b>", file=sys.stderr)
        return 2
    r = paired_delta(argv[2], argv[3])
    print(f"paired delta (B vs A), {r['n_chunks']} chunks, "
          f"{r['n_boot']} bootstrap resamples:")
    print(f"  ppl_b/ppl_a - 1 = {r['point_pct']:+.3f}%  "
          f"[95% CI {r['ci95_lo_pct']:+.3f}% .. {r['ci95_hi_pct']:+.3f}%]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
