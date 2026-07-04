#!/usr/bin/env python3
"""EF: GPTQ-style Hessian error feedback encoded directly into standard k-quant containers.

Two layers live here:

  * the encoder (ef_encode): per tensor W [nrow, n] with input Hessian H [n, n], commit
    container scales from a stock-style weighted fit, then run a GPTQ column sweep that
    quantizes to that committed two-level grid and propagates weighted error via the
    Cholesky-inverse of the dampened Hessian. act_order (process columns by descending
    diag(H), decoupled from storage order) is the ALWAYS-ON default: with magnitude-sorted
    storage order (PERM) the most important channels land last, exactly where a
    storage-order sweep dumps accumulated error (measured: catastrophic). It is exposed
    only as `unsafe_storage_order` for ablation and is never a user knob in the CLI.

  * the driver (encode_gguf): EF-encode every quantized tensor of a target GGUF and emit a
    standard GGUF that clones the target (same metadata, same per-tensor types, same byte
    layout) with tensor payloads replaced by our encodings. Weights come from the f16 GGUF.
    Composes with PERM: if perms + a spacemap are given, the f16/target must be the PERMUTED
    artifacts, and imatrix/Hessians (collected on the ORIGINAL model) are permuted here to
    match (exact -- activation statistics move with channels under permutation).

Hessians are looked up by GGUF tensor name (coldpress.hessians.load_hessian, which also
reads the old family-keyed layout). token_embd.weight has no matmul Hessian; it is upgraded
with informed column weights = the layer-0 attention input Hessian diagonal.

--device: accepted, numpy-only for now. A torch port is a marked TODO -- it must pass an
equivalence test (identical integer codes on synthetic + one real tensor vs this numpy
path) before it can become the default (contract hard rule 6).
"""
import json
import time

import numpy as np

from . import kquant as kq
from . import hessians as hess
from .ggufio import reemit, load_imatrix_means
from gguf import GGUFReader

F32 = np.float32


# ---------------------------------------------------------------- encoder core

def cholesky_inv_upper(H, damp=0.01):
    """GPTQ: returns U = chol(inv(H_damped), upper). H: [n,n] f64."""
    n = H.shape[0]
    H = H.astype(np.float64).copy()
    mean_diag = np.mean(np.diag(H))
    H[np.diag_indices(n)] += damp * mean_diag + 1e-8
    Hinv = np.linalg.inv(H)
    Hinv = (Hinv + Hinv.T) / 2
    U = np.linalg.cholesky(Hinv).T
    return np.ascontiguousarray(U)


def _grid_params(ttype, raw, nrow, n):
    """Extract per-(row, sub-block) dl, ml and value range from committed container bytes."""
    nsb = n // 256
    if ttype == "Q2_K":
        b = raw.reshape(-1, 84)
        d = b[:, 80:82].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        dmin = b[:, 82:84].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        sc = b[:, 0:16].reshape(nrow, nsb, 16)
        dl = d[..., None] * (sc & 0xF).astype(F32)
        ml = dmin[..., None] * (sc >> 4).astype(F32)
        return dl.reshape(nrow, -1), ml.reshape(nrow, -1), 0, 3, 16
    if ttype == "Q3_K":
        b = raw.reshape(-1, 110)
        d = b[:, 108:110].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        scp = b[:, 96:108].reshape(-1, 12)
        sc = (kq._unpack_scales_q3(scp) - 32).reshape(nrow, nsb, 16)
        dl = d[..., None] * sc.astype(F32)
        return dl.reshape(nrow, -1), None, -4, 3, 16
    if ttype == "Q4_K":
        b = raw.reshape(-1, 144)
        d = b[:, 0:2].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        dmin = b[:, 2:4].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        s, m = kq._unpack_scales_k4(b[:, 4:16])
        dl = d[..., None] * s.reshape(nrow, nsb, 8).astype(F32)
        ml = dmin[..., None] * m.reshape(nrow, nsb, 8).astype(F32)
        return dl.reshape(nrow, -1), ml.reshape(nrow, -1), 0, 15, 32
    if ttype == "Q5_K":
        b = raw.reshape(-1, 176)
        d = b[:, 0:2].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        dmin = b[:, 2:4].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        s, m = kq._unpack_scales_k4(b[:, 4:16])
        dl = d[..., None] * s.reshape(nrow, nsb, 8).astype(F32)
        ml = dmin[..., None] * m.reshape(nrow, nsb, 8).astype(F32)
        return dl.reshape(nrow, -1), ml.reshape(nrow, -1), 0, 31, 32
    if ttype == "Q6_K":
        b = raw.reshape(-1, 210)
        d = b[:, 208:210].copy().view(np.float16).astype(F32).reshape(nrow, nsb)
        sc = b[:, 192:208].copy().view(np.int8).astype(np.int32).reshape(nrow, nsb, 16)
        dl = d[..., None] * sc.astype(F32)
        return dl.reshape(nrow, -1), None, -32, 31, 16
    raise ValueError(ttype)


def _repack(ttype, raw, L_new, nrow, n):
    """Write new integer codes L into committed container bytes (scales unchanged)."""
    if ttype == "Q2_K":
        b = raw.reshape(-1, 84).copy()
        b[:, 16:80] = kq._pack_qs_2bit(L_new.reshape(-1, 256).astype(np.uint8))
        return b.reshape(nrow, -1)
    if ttype == "Q3_K":
        b = raw.reshape(-1, 110).copy()
        L = (L_new.reshape(-1, 256) + 4).astype(np.int32)  # signed -4..3 -> 0..7
        high = L > 3
        Lq = np.where(high, L - 4, L).astype(np.uint8)
        hmask = np.zeros((b.shape[0], 32), np.uint8)
        for j in range(8):
            hmask |= (high[:, j * 32:(j + 1) * 32].astype(np.uint8) << j)
        b[:, 0:32] = hmask
        b[:, 32:96] = kq._pack_qs_2bit(Lq)
        return b.reshape(nrow, -1)
    if ttype == "Q4_K":
        b = raw.reshape(-1, 144).copy()
        b[:, 16:144] = kq._pack_nibbles_q45(L_new.reshape(-1, 256).astype(np.uint8))
        return b.reshape(nrow, -1)
    if ttype == "Q5_K":
        b = raw.reshape(-1, 176).copy()
        L = L_new.reshape(-1, 256).astype(np.int32)
        high = (L > 15).astype(np.uint8)
        Ll = np.where(L > 15, L - 16, L).astype(np.uint8)
        qh = np.zeros((b.shape[0], 32), np.uint8)
        for c in range(4):
            qh |= high[:, c * 64:c * 64 + 32] << (2 * c)
            qh |= high[:, c * 64 + 32:c * 64 + 64] << (2 * c + 1)
        b[:, 16:48] = qh
        b[:, 48:176] = kq._pack_nibbles_q45(Ll)
        return b.reshape(nrow, -1)
    if ttype == "Q6_K":
        b = raw.reshape(-1, 210).copy()
        L = (L_new.reshape(-1, 256) + 32).astype(np.uint8)  # signed -> 0..63
        NB = b.shape[0]
        for half in range(2):
            base = half * 128
            seg = L[:, base:base + 128].reshape(NB, 4, 32)
            b[:, half * 64:half * 64 + 32] = (seg[:, 0] & 0xF) | ((seg[:, 2] & 0xF) << 4)
            b[:, half * 64 + 32:half * 64 + 64] = (seg[:, 1] & 0xF) | ((seg[:, 3] & 0xF) << 4)
            b[:, 128 + half * 32:128 + half * 32 + 32] = (
                (seg[:, 0] >> 4) | ((seg[:, 1] >> 4) << 2) |
                ((seg[:, 2] >> 4) << 4) | ((seg[:, 3] >> 4) << 6))
        return b.reshape(nrow, -1)
    raise ValueError(ttype)


def ef_encode(x, ttype, qw, H, n_iter=2, damp=0.01, act_order=True):
    """GPTQ error feedback into committed k-quant containers.
    x: [nrow, n] f32; qw: imatrix means [n] or None; H: [n, n] (X^T X).
    act_order=True: process columns by DESCENDING diag(H) (importance) regardless of
    storage order -- the committed grid fixes each column's (dl, ml) up front, so sweep
    order is free.
    Returns (raw bytes [nrow, row_bytes], recon [nrow, n])."""
    nrow, n = x.shape
    order = np.argsort(-np.diag(H)) if act_order else np.arange(n)
    Hp = H[np.ix_(order, order)]
    U = cholesky_inv_upper(Hp, damp).astype(F32)
    diagU = np.diag(U).copy()

    w_cur = x.astype(F32).copy()
    raw = None
    for it in range(n_iter):
        raw = kq.QUANTIZE[ttype](w_cur, qw)
        dl, ml, lo, hi, sbw = _grid_params(ttype, raw, nrow, n)
        wp = x.astype(F32)[:, order].copy()
        L = np.zeros((nrow, n), np.int32)
        for jj in range(n):
            j = order[jj]
            b = j // sbw
            dlj = dl[:, b]
            mlj = ml[:, b] if ml is not None else np.float32(0)
            nz = dlj != 0
            with np.errstate(divide="ignore", invalid="ignore"):
                q = kq.nearest_int((wp[:, jj] + mlj) / np.where(nz, dlj, 1))
            q = np.clip(q, lo, hi)
            q = np.where(nz, q, 0)
            L[:, j] = q
            recon_j = dlj * q.astype(F32) - mlj
            e = np.where(nz, (wp[:, jj] - recon_j) / diagU[jj], 0).astype(F32)
            if jj + 1 < n:
                wp[:, jj + 1:] -= np.outer(e, U[jj, jj + 1:])
        raw = _repack(ttype, raw, L, nrow, n)
        w_cur = np.empty_like(wp)
        w_cur[:, order] = wp
    recon = kq.DEQUANTIZE[ttype](raw, n)
    return raw, recon


# ---------------------------------------------------------------- driver

def encode_gguf(f16_path, target_path, hessdir, out_path,
                imatrix_path=None, perms=None, spacemap=None, dims=None,
                acknowledge_unreviewed=False, only="all",
                n_iter=2, unsafe_storage_order=False, device="cpu", log=print):
    """EF-encode every quantized tensor of `target_path` (a stock quant artifact) using
    the f16 weights from `f16_path` and the Hessians in `hessdir`; write a byte-map-cloned
    GGUF to `out_path`.

    perms + spacemap: if given, both the f16 and target GGUFs must already be PERMUTED, and
    this permutes the (original-order) imatrix/Hessians to match via spacemap.input_perm.
    dims/acknowledge_unreviewed: forwarded to spacemap.input_perm -- required for the
    attn_output composed index (which needs n_heads) and for the qwen35/gemma4 review gate.
    only: 'all' | 'blk' | 'embdout' for mechanism-attribution ablations.
    device: numpy-only for now (see module docstring); non-cpu logs a notice and proceeds.
    """
    if device != "cpu":
        log(f"[ef] --device {device} requested; EF is numpy-only for now, running on CPU "
            f"(torch port is a TODO gated by an equivalence test)")
    imat = load_imatrix_means(imatrix_path) if imatrix_path else {}
    fr = GGUFReader(f16_path)
    f16 = {t.name: t for t in fr.tensors}
    tr = GGUFReader(target_path)

    targets = [(t.name, t.tensor_type.name) for t in tr.tensors
               if t.tensor_type.name in kq.QUANTIZE]
    unsupported = [(t.name, t.tensor_type.name) for t in tr.tensors
                   if t.tensor_type.name not in kq.QUANTIZE
                   and t.tensor_type.name not in ("F16", "F32", "BF16")]
    if unsupported:
        raise ValueError(f"target has types we cannot encode: {unsupported}")

    ack = {"acknowledge_unreviewed": acknowledge_unreviewed} if acknowledge_unreviewed else {}

    def input_perm(name):
        if perms is None or spacemap is None:
            return None
        # dims lets the spacemap build the composed attn_output index (needs n_heads); it is
        # optional for qwen3 (inferred) but required for qwen35/gemma4.
        return spacemap.input_perm(name, perms, dims, **ack)

    replace = {}
    stats = {}
    t_start = time.time()
    for i, (name, tt) in enumerate(targets):
        is_embdout = name in ("token_embd.weight", "output.weight")
        if only == "blk" and is_embdout:
            continue
        if only == "embdout" and not is_embdout:
            continue
        x = np.asarray(f16[name].data).astype(np.float32)
        qw = imat.get(name)
        p = input_perm(name)
        if qw is not None and p is not None:
            qw = qw[p]  # imatrix stats in original order; x is already permuted
        t0 = time.time()
        if name == "token_embd.weight":
            # no matmul Hessian: informed column weights from layer-0 attn input Hessian
            H0, n0 = hess.load_hessian(hessdir, "blk.0.attn_q.weight")
            w_embd = (np.diag(H0) / max(n0, 1)).astype(np.float32)
            if p is not None:
                w_embd = w_embd[p]
            raw = kq.QUANTIZE[tt](x, w_embd)
            recon = kq.DEQUANTIZE[tt](raw, x.shape[1])
            mode = "wRTN(attn0-diag)"
        elif hess.has_hessian(hessdir, name):
            H, _n = hess.load_hessian(hessdir, name)
            if p is not None:
                H = H[np.ix_(p, p)]
            raw, recon = ef_encode(x, tt, qw, H, n_iter=n_iter,
                                   act_order=not unsafe_storage_order)
            mode = "EF" if not unsafe_storage_order else "EF(storage-order)"
        else:
            raw = kq.QUANTIZE[tt](x, qw)
            recon = kq.DEQUANTIZE[tt](raw, x.shape[1])
            mode = "wRTN"
        base = kq.roundtrip(x, tt, qw)
        w = qw if qw is not None else np.ones(x.shape[1], np.float32)
        e_base = float((((base - x) ** 2) * w).sum())
        e_ours = float((((recon - x) ** 2) * w).sum())
        stats[name] = {"type": tt, "mode": mode,
                       "rel_err_vs_stock": e_ours / e_base - 1 if e_base else 0.0,
                       "sec": time.time() - t0}
        replace[name] = raw.reshape(-1)
        if (i + 1) % 20 == 0 or is_embdout:
            log(f"[{i+1}/{len(targets)}] {name} {tt} {mode} "
                f"err-vs-stock {stats[name]['rel_err_vs_stock']*100:+.1f}% "
                f"({stats[name]['sec']:.1f}s)")
        del x

    reemit(target_path, out_path, replace=replace)
    if stats:
        med = float(np.median([s["rel_err_vs_stock"] for s in stats.values()]))
        log(f"median err-vs-stock: {med*100:+.2f}%; total {(time.time()-t_start)/60:.1f} min")
    json.dump(stats, open(out_path + ".efstats.json", "w"), indent=1)
    return stats
