#!/usr/bin/env python3
"""The exact PERM objective: imatrix-weighted squared reconstruction error of the FAITHFUL
llama.cpp encoder mimic (coldpress.kquant) at a tensor's TARGET k-quant type.

err(P) = sum_c qw[c] * sum_r (x_hat[r,c] - x[r,c])^2, computed on x[:,P], qw[P].
Tensors without imatrix entries (token_embd/output) use the ref (unweighted) path and
uniform error weights -- exactly what stock llama-quantize does to them.

This module is architecture-agnostic. Spacemaps call tensor_err / col_log_rms to score
candidate permutations of the spaces they declare.
"""
import numpy as np

from .. import kquant as kq


def tensor_err(x, ttype, qw, perm=None, wvec=None, rows_sample=None, seed=0):
    """Weighted quant error of x under permutation `perm` at container `ttype`."""
    if ttype in ("F16", "F32", "BF16"):
        return 0.0
    if rows_sample and x.shape[0] > rows_sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], rows_sample, replace=False))
        x = x[idx]
    x = np.asarray(x, dtype=np.float32)
    if perm is not None:
        x = x[:, perm]
        qw = qw[perm] if qw is not None else None
    if ttype not in kq.QUANTIZE:
        raise ValueError(f"no encoder for {ttype}")
    recon = kq.roundtrip(np.ascontiguousarray(x), ttype, qw)
    d2 = (recon.astype(np.float64) - x.astype(np.float64)) ** 2
    colerr = d2.sum(0)
    if wvec is not None:
        w = wvec[perm] if perm is not None else wvec
        return float((colerr * w).sum())
    if qw is not None:
        return float((colerr * qw).sum())
    return float(colerr.sum())


def col_log_rms(x, row_step=32768):
    """Per-column log-RMS profile (a candidate sort key)."""
    x = np.asarray(x)
    acc = np.zeros(x.shape[1], np.float64)
    for r0 in range(0, x.shape[0], row_step):
        xc = np.asarray(x[r0:r0 + row_step], dtype=np.float64)
        acc += (xc ** 2).sum(0)
    return np.log(np.sqrt(acc / x.shape[0]) + 1e-12)
