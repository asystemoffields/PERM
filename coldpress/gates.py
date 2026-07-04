#!/usr/bin/env python3
"""The verification gates. Each encodes a measured failure; the hard ones fail loudly.

  g3                  logits equality of the permuted model vs original (rel < 1e-4). HARD.
  byte_parity         final GGUF's per-tensor type+size == stock map, total <= stock. HARD.
  encoder_faithfulness  our stock-path encoder vs real llama-quantize bytes (needs a stock
                      artifact from llama.cpp; skippable): median byte-diff < 0.1%.
  smoke               llama-cli loads the artifact and generates coherently.
"""
import numpy as np

from . import kquant as kq
from .ggufio import read_typemap, load_imatrix_means
from gguf import GGUFReader


class GateError(AssertionError):
    pass


def g3(spacemap, model, perms, dims, ids, threshold=1e-4, acknowledge_unreviewed=None):
    """Run the spacemap's G3 logits-equality gate. HARD-fails if rel >= threshold.
    Returns (max_abs_dlogit, rel)."""
    kwargs = {}
    if acknowledge_unreviewed is not None:
        kwargs["acknowledge_unreviewed"] = acknowledge_unreviewed
    d, rel = spacemap.g3_check(model, perms, dims, ids, **kwargs)
    if not (rel < threshold):
        raise GateError(f"G3 FAIL: rel logits drift {rel:.3e} >= {threshold:.0e} "
                        f"(max|dlogit|={d:.3e}) -- permutation is not function-preserving")
    return d, rel


def byte_parity(final_gguf, stock_gguf):
    """Assert the final artifact matches the stock quant's per-tensor type map and does not
    exceed its total tensor bytes. HARD-fails on mismatch. Returns a report dict."""
    fm = read_typemap(final_gguf)
    sm = read_typemap(stock_gguf)
    mismatches = []
    if set(fm) != set(sm):
        only_final = sorted(set(fm) - set(sm))
        only_stock = sorted(set(sm) - set(fm))
        raise GateError(f"byte-parity FAIL: tensor-set differs "
                        f"(+final {only_final[:4]}, +stock {only_stock[:4]})")
    for name in sm:
        if fm[name]["type"] != sm[name]["type"]:
            mismatches.append((name, "type", fm[name]["type"], sm[name]["type"]))
        if fm[name]["n_bytes"] != sm[name]["n_bytes"]:
            mismatches.append((name, "n_bytes", fm[name]["n_bytes"], sm[name]["n_bytes"]))
    total_final = sum(v["n_bytes"] for v in fm.values())
    total_stock = sum(v["n_bytes"] for v in sm.values())
    if mismatches:
        raise GateError(f"byte-parity FAIL: {len(mismatches)} per-tensor mismatches, "
                        f"e.g. {mismatches[:3]}")
    if total_final > total_stock:
        raise GateError(f"byte-parity FAIL: total {total_final} > stock {total_stock}")
    return {"tensors": len(sm), "total_final": total_final, "total_stock": total_stock,
            "passed": True}


def encoder_faithfulness(f16_gguf, stock_quant_gguf, imatrix_path=None,
                         max_byte_diff=0.001, max_err_delta=0.01, max_tensors=999, log=print):
    """Differential test: re-encode the f16 weights with our stock-path encoder and compare
    byte-for-byte to a REAL llama-quantize artifact (stock_quant_gguf, produced by
    llama.cpp). Numpy pairwise summation may flip rare near-tie scale acceptances, so we
    require median byte-diff < max_byte_diff and |err delta| < max_err_delta, not equality.
    Returns (passed, report)."""
    fr = GGUFReader(f16_gguf)
    qr = GGUFReader(stock_quant_gguf)
    f16 = {t.name: t for t in fr.tensors}
    imat = load_imatrix_means(imatrix_path) if imatrix_path else {}
    worst = []
    tested = 0
    for t in qr.tensors:
        tt = t.tensor_type.name
        if tt not in kq.QUANTIZE or tested >= max_tensors:
            continue
        qw = imat.get(t.name)
        theirs = np.ascontiguousarray(np.asarray(t.data))
        src = np.asarray(f16[t.name].data)
        nrow = src.shape[0]
        step = max(1, min(nrow, (1 << 24) // src.shape[1]))
        n_diff = 0
        e_ours = e_theirs = 0.0
        w = qw if qw is not None else np.ones(src.shape[1], np.float32)
        for r0 in range(0, nrow, step):
            x = src[r0:r0 + step].astype(np.float32)
            ours = kq.QUANTIZE[tt](x, qw)
            th = theirs[r0:r0 + step]
            assert ours.nbytes == th.nbytes, (t.name, ours.nbytes, th.nbytes)
            n_diff += int((ours != th).sum())
            d_ours = kq.DEQUANTIZE[tt](ours, x.shape[1])
            d_theirs = kq.DEQUANTIZE[tt](np.ascontiguousarray(th), x.shape[1])
            e_ours += float((((d_ours - x) ** 2) * w).sum())
            e_theirs += float((((d_theirs - x) ** 2) * w).sum())
        frac = n_diff / theirs.nbytes
        rel = e_ours / e_theirs - 1 if e_theirs else 0.0
        worst.append((frac, rel, t.name, tt))
        tested += 1
    worst.sort(reverse=True)
    med = float(np.median([f for f, *_ in worst])) if worst else 0.0
    mre = max((abs(r) for _, r, *_ in worst), default=0.0)
    passed = med < max_byte_diff and mre < max_err_delta
    for frac, rel, name, tt in worst[:8]:
        log(f"  byte-diff={frac*100:6.3f}%  err-vs-theirs={rel*100:+7.3f}%  {name} [{tt}]")
    log(f"encoder faithfulness: median byte-diff {med*100:.3f}%, max|err delta| {mre*100:.3f}% "
        f"-> {'PASS' if passed else 'FAIL'}")
    return passed, {"median_byte_diff": med, "max_err_delta": mre, "tested": tested}


def smoke(llama, model_gguf, prompt="The capital of France is", n_predict=48, ngl=0,
          logfile=None):
    """Load the artifact in llama-cli and generate. Returns the generated text; asserts the
    model loaded and produced non-empty output."""
    out = llama.cli(model_gguf, prompt, n_predict=n_predict, ngl=ngl, logfile=logfile)
    body = out[len(prompt):] if prompt in out else out
    assert body.strip(), "smoke FAIL: llama-cli produced no generation"
    return out
