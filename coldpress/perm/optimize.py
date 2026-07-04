#!/usr/bin/env python3
"""PERM driver: read the f16 GGUF + target typemap + imatrix, run the spacemap optimizer.

Kept thin -- the arch-specific space structure and objective live in the spacemap; this
module just marshals GGUF I/O into the {weights, ttypes, qws} accessor the spacemap wants.
"""
import numpy as np

from ..ggufio import load_imatrix_means, read_typemap
from gguf import GGUFReader


def run_perm(spacemap, f16_path, typemap, dims, imatrix_path=None, rows_sample=16384,
             log=print, **kwargs):
    """Return (perms, report). typemap: {gguf_name: {'type': ttype, ...}}.

    **kwargs (e.g. acknowledge_unreviewed) are forwarded to the spacemap's optimize; empty
    for the reviewed maps (qwen3), {'acknowledge_unreviewed': True} for the gated ones."""
    r = GGUFReader(f16_path)
    weights = {t.name: np.asarray(t.data) for t in r.tensors}
    ttypes = {name: typemap[name]["type"] for name in typemap}
    qws = load_imatrix_means(imatrix_path) if imatrix_path else {}
    perms, report = spacemap.optimize(weights, ttypes, qws, dims,
                                      rows_sample=rows_sample, log=log, **kwargs)
    return perms, report


def load_target_typemap(target_path):
    return read_typemap(target_path)
