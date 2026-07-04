#!/usr/bin/env python3
"""Vectorized numpy reimplementation of ggml k-quant containers (llama.cpp @ 039e20a2).

Faithful to /data/coldpress/results/kquant-reference.md:
- decoders: bit-exact.
- encoders: algorithm-faithful, vectorized across blocks. numpy pairwise summation may
  flip rare near-tie scale acceptances vs C's sequential fp32 loops -> not guaranteed
  bit-identical to llama-quantize, but statistically equivalent (validated differentially).

Weight matrices: x is [nrow, n_per_row] float32; qw (imatrix means) is [n_per_row] or None.
The SAME qw vector applies to every row (llama.cpp does not advance it across rows).
ref path (qw=None) == what llama-quantize does for tensors without imatrix entries
(token_embd.weight / output.weight!); impl path == with imatrix.
"""
import numpy as np

F32 = np.float32
GROUP_MAX_EPS = np.float32(1e-15)
QK_K = 256

BYTES_PER_SB = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
SUB_W = {"Q2_K": 16, "Q3_K": 16, "Q4_K": 32, "Q5_K": 32, "Q6_K": 16}


def nearest_int(x):
    """ggml nearest_int: round-half-even via fp32 magic add. x: f32 array -> int32."""
    x = np.ascontiguousarray(x, dtype=F32)
    return ((x + np.float32(12582912.0)).view(np.int32) & 0x007FFFFF) - 0x00400000


def fp16rt(x):
    """fp32 -> fp16 -> fp32 round trip (part of every quantizer)."""
    return np.asarray(x, dtype=F32).astype(np.float16).astype(F32)


def _f16bits(x):
    return np.asarray(x, dtype=F32).astype(np.float16)


# ---------------------------------------------------------------- helpers

def make_qkx_quants(x, weights, nmax, rmin, rdelta, nstep, use_mad, qkx3):
    """make_qkx2_quants / make_qkx3_quants, vectorized over blocks.
    x, weights: [NB, n] f32. Returns (scale [NB], the_min [NB], L [NB,n] uint8)."""
    x = x.astype(F32)
    w = weights.astype(F32)
    NB, n = x.shape
    mn = x.min(1)
    mx = x.max(1)
    sum_w = w.sum(1, dtype=F32)
    sum_x = (w * x).sum(1, dtype=F32)
    mn = np.minimum(mn, np.float32(0))
    dead = (mx <= mn) if qkx3 else (mx == mn)
    span = np.where(dead, np.float32(1), mx - mn)  # avoid div0; dead blocks masked at end

    iscale = np.float32(nmax) / span
    scale = np.float32(1) / iscale
    L = np.clip(nearest_int(iscale[:, None] * (x - mn[:, None])), 0, nmax)
    diff = scale[:, None] * L.astype(F32) + mn[:, None] - x
    diff = np.abs(diff) if use_mad else diff * diff
    best = (w * diff).sum(1, dtype=F32)

    for step in range(nstep + 1):
        # C recomputes the denominator from the RUNNING min each candidate (min is
        # mutated on acceptance); numerator and denominator must stay consistent.
        span_cur = mx - mn
        span_cur = np.where(span_cur > 0, span_cur, np.float32(1))
        isc = (np.float32(rmin) + np.float32(rdelta) * np.float32(step) + np.float32(nmax)) / span_cur
        laux = np.clip(nearest_int(isc[:, None] * (x - mn[:, None])), 0, nmax)
        lf = laux.astype(F32)
        sum_l = (w * lf).sum(1, dtype=F32)
        sum_l2 = (w * lf * lf).sum(1, dtype=F32)
        sum_xl = (w * lf * x).sum(1, dtype=F32)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
            alt_scale = sum_xl / sum_l2
        pos = this_min > 0
        this_min = np.where(pos, np.float32(0), this_min)
        this_scale = np.where(pos, alt_scale, this_scale)
        diff = this_scale[:, None] * lf + this_min[:, None] - x
        diff = np.abs(diff) if use_mad else diff * diff
        cur = (w * diff).sum(1, dtype=F32)
        acc = ok & (cur < best) & ~dead
        L = np.where(acc[:, None], laux, L)
        best = np.where(acc, cur, best)
        scale = np.where(acc, this_scale, scale)
        mn = np.where(acc, this_min, mn)

    scale = np.where(dead, np.float32(0), scale)
    L = np.where(dead[:, None], 0, L)
    return scale.astype(F32), (-mn).astype(F32), L.astype(np.uint8)


def make_qx_quants(x, qw, nmax):
    """make_qx_quants rmse_type=1, vectorized. x [NB,n]; qw [NB,n] or None (w=x*x).
    Returns (scale [NB], L [NB,n] int32 offset +nmax)."""
    x = x.astype(F32)
    NB, n = x.shape
    ax = np.abs(x)
    idx = ax.argmax(1)
    amax = ax[np.arange(NB), idx]
    mx = x[np.arange(NB), idx]  # signed value of extreme element
    dead = amax < GROUP_MAX_EPS
    mx = np.where(dead, np.float32(1), mx)
    w = (qw if qw is not None else x * x).astype(F32)

    def pass_(iscale):
        l = np.clip(nearest_int(iscale[:, None] * x), -nmax, nmax - 1)
        lf = l.astype(F32)
        sumlx = (w * x * lf).sum(1, dtype=F32)
        suml2 = (w * lf * lf).sum(1, dtype=F32)
        return l, sumlx, suml2

    iscale = -np.float32(nmax) / mx
    L, sumlx, suml2 = pass_(iscale)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(suml2 != 0, sumlx / suml2, np.float32(0))
    best = scale * sumlx
    for s in list(range(-9, 0)) + list(range(1, 10)):
        isc = -(np.float32(nmax) + np.float32(0.1) * np.float32(s)) / mx
        l, sumlx_, suml2_ = pass_(isc)
        acc = (suml2_ > 0) & (sumlx_ * sumlx_ > best * suml2_) & ~dead
        with np.errstate(divide="ignore", invalid="ignore"):
            newscale = sumlx_ / suml2_
        L = np.where(acc[:, None], l, L)
        scale = np.where(acc, newscale, scale)
        best = np.where(acc, newscale * sumlx_, best)
    scale = np.where(dead, np.float32(0), scale)
    L = np.where(dead[:, None], 0, L)
    return scale.astype(F32), (L + nmax).astype(np.int32)


def make_q3_quants(x, nmax=4):
    """make_q3_quants do_rmse=true (q3_K ref path). x [NB,16]. w = x*x hardcoded.
    Sequential 5-sweep coordinate descent, vectorized across blocks."""
    x = x.astype(F32)
    NB, n = x.shape
    ax = np.abs(x)
    idx = ax.argmax(1)
    amax = ax[np.arange(NB), idx]
    mx = x[np.arange(NB), idx]
    dead = amax < GROUP_MAX_EPS
    mx = np.where(dead, np.float32(1), mx)
    iscale = -np.float32(nmax) / mx
    L = np.clip(nearest_int(iscale[:, None] * x), -nmax, nmax - 1).astype(F32)
    w = x * x
    sumlx = (w * x * L).sum(1, dtype=F32)
    suml2 = (w * L * L).sum(1, dtype=F32)
    active = ~dead
    for _ in range(5):
        changed = np.zeros(NB, bool)
        for i in range(n):
            wi = w[:, i]
            xi = x[:, i]
            Li = L[:, i]
            slx = sumlx - wi * xi * Li
            cond = (slx > 0) & active
            sl2 = suml2 - wi * Li * Li
            with np.errstate(divide="ignore", invalid="ignore"):
                new_l = np.clip(nearest_int(xi * sl2 / np.where(slx == 0, np.float32(1), slx)),
                                -nmax, nmax - 1).astype(F32)
            slx2 = slx + wi * xi * new_l
            sl22 = sl2 + wi * new_l * new_l
            acc = cond & (new_l != Li) & (sl22 > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            L[:, i] = np.where(acc, new_l, Li)
            sumlx = np.where(acc, slx2, sumlx)
            suml2 = np.where(acc, sl22, suml2)
            changed |= acc
        active &= changed
        if not active.any():
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(suml2 > 0, sumlx / suml2, np.float32(0))
    scale = np.where(dead, np.float32(0), scale)
    L = np.where(dead[:, None], np.float32(0), L)
    return scale.astype(F32), (L.astype(np.int32) + nmax)


def make_qp_quants(x, sw, nmax):
    """make_qp_quants (nonnegative, for super-scales). x, sw: [NB, n]. Returns (d [NB], L [NB,n] uint8)."""
    x = x.astype(F32)
    w = sw.astype(F32)
    NB, n = x.shape
    mx = x.max(1)
    dead = mx < GROUP_MAX_EPS
    mxs = np.where(dead, np.float32(1), mx)
    iscale = np.float32(nmax) / mxs
    # initial best_mse uses UNCLAMPED L
    L0 = nearest_int(iscale[:, None] * x).astype(F32)
    scale = np.float32(1) / iscale
    diff = x - scale[:, None] * L0
    best_mse = (w * diff * diff).sum(1, dtype=F32)
    for s in list(range(-4, 0)) + list(range(1, 5)):
        isc = (np.float32(0.1) * np.float32(s) + np.float32(nmax)) / mxs
        sc = np.float32(1) / isc
        l = np.minimum(nearest_int(isc[:, None] * x), nmax).astype(F32)
        diff = x - sc[:, None] * l
        mse = (w * diff * diff).sum(1, dtype=F32)
        acc = (mse < best_mse) & ~dead
        best_mse = np.where(acc, mse, best_mse)
        iscale = np.where(acc, isc, iscale)
    L = np.minimum(nearest_int(iscale[:, None] * x), nmax).astype(F32)
    sumlx = (w * x * L).sum(1, dtype=F32)
    suml2 = (w * L * L).sum(1, dtype=F32)
    active = ~dead
    for _ in range(5):
        changed = np.zeros(NB, bool)
        for i in range(n):
            wi = w[:, i]
            xi = x[:, i]
            Li = L[:, i]
            slx = sumlx - wi * xi * Li
            sl2 = suml2 - wi * Li * Li
            cond = (slx > 0) & (sl2 > 0) & active
            with np.errstate(divide="ignore", invalid="ignore"):
                new_l = np.minimum(nearest_int(xi * sl2 / np.where(slx == 0, np.float32(1), slx)),
                                   nmax).astype(F32)
            slx2 = slx + wi * xi * new_l
            sl22 = sl2 + wi * new_l * new_l
            acc = cond & (new_l != Li) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            L[:, i] = np.where(acc, new_l, Li)
            sumlx = np.where(acc, slx2, sumlx)
            suml2 = np.where(acc, sl22, suml2)
            changed |= acc
        active &= changed
        if not active.any():
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(suml2 > 0, sumlx / suml2, np.float32(0))
    d = np.where(dead, np.float32(0), d)
    L = np.where(dead[:, None], np.float32(0), L)
    return d.astype(F32), L.astype(np.uint8)


# ---------------------------------------------------------------- packing utils

def _pack_qs_2bit(L):
    """q2_K/q3_K low-2-bit packing. L: [NB,256] uint8 (values 0..3) -> [NB,64] uint8."""
    NB = L.shape[0]
    out = np.zeros((NB, 64), np.uint8)
    for half in range(2):  # two 128-chunks
        base = half * 128
        seg = L[:, base:base + 128].reshape(NB, 4, 32)  # [l], [l+32], [l+64], [l+96]
        out[:, half * 32:(half + 1) * 32] = (seg[:, 0] | (seg[:, 1] << 2) |
                                             (seg[:, 2] << 4) | (seg[:, 3] << 6))
    return out


def _unpack_qs_2bit(qs):
    """inverse of _pack_qs_2bit. qs: [NB,64] -> [NB,256] uint8 0..3."""
    NB = qs.shape[0]
    L = np.zeros((NB, 256), np.uint8)
    for half in range(2):
        b = qs[:, half * 32:(half + 1) * 32]
        base = half * 128
        for k in range(4):
            L[:, base + 32 * k: base + 32 * (k + 1)] = (b >> (2 * k)) & 3
    return L


def _pack_scales_q3(Ls):
    """q3_K 6-bit scale packing. Ls: [NB,16] ints 0..63 -> [NB,12] uint8."""
    NB = Ls.shape[0]
    out = np.zeros((NB, 12), np.uint8)
    l = Ls.astype(np.uint8)
    for j in range(16):
        lj = l[:, j]
        if j < 8:
            out[:, j] |= lj & 0xF
        else:
            out[:, j - 8] |= (lj & 0xF) << 4
        out[:, 8 + j % 4] |= (lj >> 4) << (2 * (j // 4))
    return out


def _unpack_scales_q3(sc):
    """[NB,12] -> [NB,16] int32 (0..63, subtract 32 for signed)."""
    NB = sc.shape[0]
    out = np.zeros((NB, 16), np.int32)
    for j in range(16):
        if j < 8:
            lo = sc[:, j] & 0xF
        else:
            lo = sc[:, j - 8] >> 4
        hi = (sc[:, 8 + j % 4] >> (2 * (j // 4))) & 3
        out[:, j] = lo | (hi << 4)
    return out


def _pack_scales_k4(Ls, Lm):
    """q4_K/q5_K 6-bit scale/min packing. Ls, Lm: [NB,8] 0..63 -> [NB,12] uint8."""
    NB = Ls.shape[0]
    out = np.zeros((NB, 12), np.uint8)
    ls = Ls.astype(np.uint8)
    lm = Lm.astype(np.uint8)
    for j in range(8):
        if j < 4:
            out[:, j] = ls[:, j]
            out[:, j + 4] = lm[:, j]
        else:
            out[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
            out[:, j - 4] |= (ls[:, j] >> 4) << 6
            out[:, j] |= (lm[:, j] >> 4) << 6
    return out


def _unpack_scales_k4(sc):
    """[NB,12] -> (scales [NB,8], mins [NB,8]) int32 0..63 (get_scale_min_k4)."""
    NB = sc.shape[0]
    d = np.zeros((NB, 8), np.int32)
    m = np.zeros((NB, 8), np.int32)
    for j in range(8):
        if j < 4:
            d[:, j] = sc[:, j] & 63
            m[:, j] = sc[:, j + 4] & 63
        else:
            d[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            m[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return d, m


def _pack_nibbles_q45(L):
    """q4_K/q5_K low-nibble packing: 64-chunks, (l, l+32). L: [NB,256] -> [NB,128]."""
    NB = L.shape[0]
    out = np.zeros((NB, 128), np.uint8)
    for c in range(4):
        seg = L[:, c * 64:(c + 1) * 64]
        out[:, c * 32:(c + 1) * 32] = (seg[:, :32] & 0xF) | ((seg[:, 32:] & 0xF) << 4)
    return out


def _unpack_nibbles_q45(qs):
    NB = qs.shape[0]
    L = np.zeros((NB, 256), np.uint8)
    for c in range(4):
        b = qs[:, c * 32:(c + 1) * 32]
        L[:, c * 64:c * 64 + 32] = b & 0xF
        L[:, c * 64 + 32:c * 64 + 64] = b >> 4
    return L


# ---------------------------------------------------------------- q2_K

def quantize_q2_K(x, qw=None):
    """x: [nrow, n_per_row] f32; qw: [n_per_row] f32 imatrix means or None.
    Returns raw bytes [nrow, n_per_row//256 * 84] uint8."""
    nrow, npr = x.shape
    assert npr % QK_K == 0
    xb = x.reshape(-1, QK_K).astype(F32)  # [NB, 256]
    NB = xb.shape[0]
    xs = xb.reshape(NB, 16, 16)

    if qw is None:
        # ref path: weights=|x|, make_qkx2(16,3,-0.5,0.1,15,mad=true); superscale q4scale=15
        w = np.abs(xs)
        scale, the_min, L = make_qkx_quants(xs.reshape(-1, 16), w.reshape(-1, 16),
                                            3, -0.5, 0.1, 15, True, False)
        scales = scale.reshape(NB, 16)
        mins = the_min.reshape(NB, 16)
        L = L.reshape(NB, 256)
        max_scale = scales.max(1)
        max_min = mins.max(1)
        sc_i = np.zeros((NB, 16), np.int32)
        pos = max_scale > 0
        isc = np.where(pos, np.float32(15) / np.where(pos, max_scale, 1), np.float32(0))
        sc_i = np.where(pos[:, None], nearest_int(isc[:, None] * scales), 0)
        d = np.where(pos, max_scale / np.float32(15), np.float32(0))
        posm = max_min > 0
        iscm = np.where(posm, np.float32(15) / np.where(posm, max_min, 1), np.float32(0))
        mn_i = np.where(posm[:, None], nearest_int(iscm[:, None] * mins), 0)
        dmin = np.where(posm, max_min / np.float32(15), np.float32(0))
        d16 = _f16bits(d)
        dmin16 = _f16bits(dmin)
        dl = d16.astype(F32)[:, None] * sc_i.astype(F32)
        ml = dmin16.astype(F32)[:, None] * mn_i.astype(F32)
    else:
        qwb = np.broadcast_to(np.asarray(qw, F32), (nrow, npr)).reshape(-1, QK_K)
        sumx2 = (xb * xb).sum(1, dtype=F32)
        sigma2 = sumx2 / np.float32(QK_K)
        weight = qwb.reshape(NB, 16, 16) * np.sqrt(sigma2[:, None, None] + xs * xs)
        sw = weight.sum(2, dtype=F32)  # [NB,16]
        scale, the_min, L = make_qkx_quants(xs.reshape(-1, 16), weight.reshape(-1, 16),
                                            3, -0.9, 0.05, 36, False, True)
        scales = scale.reshape(NB, 16)
        mins = the_min.reshape(NB, 16)
        L = L.reshape(NB, 256)
        dm, Ls = make_qp_quants(scales, sw, 15)
        mm, Lm = make_qp_quants(mins, sw, 15)
        sc_i = Ls.astype(np.int32)
        mn_i = Lm.astype(np.int32)
        d16 = _f16bits(dm)
        dmin16 = _f16bits(mm)
        dl = d16.astype(F32)[:, None] * sc_i.astype(F32)
        ml = dmin16.astype(F32)[:, None] * mn_i.astype(F32)

    # requantize where dl != 0 (stale L kept where dl == 0)
    dl_e = np.repeat(dl, 16, axis=1)
    ml_e = np.repeat(ml, 16, axis=1)
    nz = dl_e != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.clip(nearest_int((xb + ml_e) / np.where(nz, dl_e, 1)), 0, 3).astype(np.uint8)
    L = np.where(nz, lq, L)

    scales_b = (sc_i.astype(np.uint8) & 0xF) | ((mn_i.astype(np.uint8) & 0xF) << 4)
    out = np.zeros((NB, 84), np.uint8)
    out[:, 0:16] = scales_b
    out[:, 16:80] = _pack_qs_2bit(L)
    out[:, 80:82] = d16.view(np.uint8).reshape(NB, 2)
    out[:, 82:84] = dmin16.view(np.uint8).reshape(NB, 2)
    return out.reshape(nrow, -1)


def dequantize_q2_K(raw, n_per_row):
    nrow = raw.shape[0]
    b = raw.reshape(-1, 84)
    NB = b.shape[0]
    d = b[:, 80:82].copy().view(np.float16).astype(F32).reshape(NB)
    dmin = b[:, 82:84].copy().view(np.float16).astype(F32).reshape(NB)
    sc = b[:, 0:16]
    L = _unpack_qs_2bit(b[:, 16:80]).astype(F32)
    dl = d[:, None] * (sc & 0xF).astype(F32)
    ml = dmin[:, None] * (sc >> 4).astype(F32)
    y = np.repeat(dl, 16, 1) * L - np.repeat(ml, 16, 1)
    return y.reshape(nrow, n_per_row)


# ---------------------------------------------------------------- q3_K

def quantize_q3_K(x, qw=None):
    nrow, npr = x.shape
    xb = x.reshape(-1, QK_K).astype(F32)
    NB = xb.shape[0]
    xs = xb.reshape(NB, 16, 16)

    if qw is None:
        scale, L = make_q3_quants(xs.reshape(-1, 16), 4)
        scales = scale.reshape(NB, 16)
        L = L.reshape(NB, 256)
        amax_idx = np.abs(scales).argmax(1)
        max_scale = scales[np.arange(NB), amax_idx]
        nz = np.abs(scales).max(1) > 0  # max_scale != 0 test in C is on the signed value
        nz = max_scale != 0
        isc = np.where(nz, np.float32(-32) / np.where(nz, max_scale, 1), np.float32(0))
        Ls = np.where(nz[:, None],
                      np.clip(nearest_int(isc[:, None] * scales), -32, 31) + 32, 0)
        d = np.where(nz, np.float32(1) / isc, np.float32(0))
        d = np.where(nz, np.where(np.isfinite(d), d, 0), 0)
    else:
        qwb = np.broadcast_to(np.asarray(qw, F32), (nrow, npr)).reshape(-1, QK_K)
        sumx2 = (xb * xb).sum(1, dtype=F32)
        sigma2 = np.float32(2) * sumx2 / np.float32(QK_K)
        weight = qwb.reshape(NB, 16, 16) * np.sqrt(sigma2[:, None, None] + xs * xs)
        sw = weight.sum(2, dtype=F32)
        scale, L = make_qx_quants(xs.reshape(-1, 16), weight.reshape(-1, 16), 4)
        scales = scale.reshape(NB, 16)
        L = L.reshape(NB, 256)
        d, Ls = make_qx_quants(scales, sw, 32)  # Ls already offset +32 (0..63)

    L = L.astype(np.int32)  # 0..7 (offset +4)
    scales_packed = _pack_scales_q3(Ls.astype(np.int32))
    d16 = _f16bits(d)
    # requantize: sc = unpacked - 32
    sc_unpacked = _unpack_scales_q3(scales_packed) - 32
    dsub = d16.astype(F32)[:, None] * sc_unpacked.astype(F32)
    dsub_e = np.repeat(dsub, 16, axis=1)
    nz = dsub_e != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.clip(nearest_int(xb / np.where(nz, dsub_e, 1)), -4, 3) + 4
    L = np.where(nz, lq, L)

    # hmask: bit set where L > 3, then L -= 4
    high = L > 3
    Lq = np.where(high, L - 4, L).astype(np.uint8)
    hmask = np.zeros((NB, 32), np.uint8)
    for j in range(8):  # bit j covers elements j*32..(j+1)*32, at hmask[element % 32]
        seg = high[:, j * 32:(j + 1) * 32]
        hmask |= (seg.astype(np.uint8) << j)
    out = np.zeros((NB, 110), np.uint8)
    out[:, 0:32] = hmask
    out[:, 32:96] = _pack_qs_2bit(Lq)
    out[:, 96:108] = scales_packed
    out[:, 108:110] = d16.view(np.uint8).reshape(NB, 2)
    return out.reshape(nrow, -1)


def dequantize_q3_K(raw, n_per_row):
    nrow = raw.shape[0]
    b = raw.reshape(-1, 110)
    NB = b.shape[0]
    d = b[:, 108:110].copy().view(np.float16).astype(F32).reshape(NB)
    sc = _unpack_scales_q3(b[:, 96:108]) - 32  # [NB,16]
    lo = _unpack_qs_2bit(b[:, 32:96]).astype(np.int32)
    hmask = b[:, 0:32]
    hi = np.zeros((NB, 256), np.int32)
    for j in range(8):
        hi[:, j * 32:(j + 1) * 32] = (hmask >> j) & 1
    q = lo - np.where(hi == 0, 4, 0)  # cleared bit subtracts 4
    dl = d[:, None] * sc.astype(F32)
    y = np.repeat(dl, 16, 1) * q.astype(F32)
    return y.reshape(nrow, n_per_row)


# ---------------------------------------------------------------- q4_K / q5_K

def _quantize_q45_K(x, qw, nmax, is_q5):
    nrow, npr = x.shape
    xb = x.reshape(-1, QK_K).astype(F32)
    NB = xb.shape[0]
    xs = xb.reshape(NB, 8, 32)

    if qw is None:
        sum_x2 = (xs * xs).sum(2, dtype=F32)
        av_x = np.sqrt(sum_x2 / np.float32(32))
        w = av_x[:, :, None] + np.abs(xs)
        rmin, rdelta, nstep = (-0.5, 0.1, 15) if is_q5 else (-1.0, 0.1, 20)
        scale, the_min, L = make_qkx_quants(xs.reshape(-1, 32), w.reshape(-1, 32),
                                            nmax, rmin, rdelta, nstep, False, False)
        scales = scale.reshape(NB, 8)
        mins = the_min.reshape(NB, 8)
        L = L.reshape(NB, 256)
        max_scale = scales.max(1)
        max_min = mins.max(1)
        inv_scale = np.where(max_scale > 0, np.float32(63) / np.where(max_scale > 0, max_scale, 1),
                             np.float32(0))
        inv_min = np.where(max_min > 0, np.float32(63) / np.where(max_min > 0, max_min, 1),
                           np.float32(0))
        # uint8 truncation BEFORE MIN(63,...) clamp (wrap-then-clamp)
        Ls = np.minimum(nearest_int(inv_scale[:, None] * scales).astype(np.uint8), 63)
        Lm = np.minimum(nearest_int(inv_min[:, None] * mins).astype(np.uint8), 63)
        d = max_scale / np.float32(63)
        dmin = max_min / np.float32(63)
    else:
        qwb = np.broadcast_to(np.asarray(qw, F32), (nrow, npr)).reshape(-1, QK_K)
        sum_x2 = (xb * xb).sum(1, dtype=F32)
        sigma2 = np.float32(2) * sum_x2 / np.float32(QK_K)
        w = qwb.reshape(NB, 8, 32) * np.sqrt(sigma2[:, None, None] + xs * xs)
        sw = w.sum(2, dtype=F32)
        scale, the_min, L = make_qkx_quants(xs.reshape(-1, 32), w.reshape(-1, 32),
                                            nmax, -0.9, 0.05, 36, False, True)
        scales = scale.reshape(NB, 8)
        mins = the_min.reshape(NB, 8)
        L = L.reshape(NB, 256)
        d, Ls = make_qp_quants(scales, sw, 63)
        dmin, Lm = make_qp_quants(mins, sw, 63)
        if is_q5:
            Ls = np.minimum(Ls, 63)
            Lm = np.minimum(Lm, 63)

    d16 = _f16bits(d)
    dmin16 = _f16bits(dmin)
    scales_packed = _pack_scales_k4(Ls, Lm)
    sc_u, mn_u = _unpack_scales_k4(scales_packed)
    dsub = d16.astype(F32)[:, None] * sc_u.astype(F32)
    msub = dmin16.astype(F32)[:, None] * mn_u.astype(F32)
    dsub_e = np.repeat(dsub, 32, 1)
    msub_e = np.repeat(msub, 32, 1)
    nz = dsub_e != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.clip(nearest_int((xb + msub_e) / np.where(nz, dsub_e, 1)), 0, nmax)
    L = np.where(nz, lq.astype(np.uint8), L)

    if not is_q5:
        out = np.zeros((NB, 144), np.uint8)
        out[:, 0:2] = d16.view(np.uint8).reshape(NB, 2)
        out[:, 2:4] = dmin16.view(np.uint8).reshape(NB, 2)
        out[:, 4:16] = scales_packed
        out[:, 16:144] = _pack_nibbles_q45(L)
        return out.reshape(nrow, -1)
    # q5: strip high bit into qh bit-planes
    high = (L > 15).astype(np.uint8)
    Ll = np.where(L > 15, L - 16, L).astype(np.uint8)
    qh = np.zeros((NB, 32), np.uint8)
    for c in range(4):
        qh |= high[:, c * 64:c * 64 + 32] << (2 * c)       # m1
        qh |= high[:, c * 64 + 32:c * 64 + 64] << (2 * c + 1)  # m2
    out = np.zeros((NB, 176), np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(NB, 2)
    out[:, 2:4] = dmin16.view(np.uint8).reshape(NB, 2)
    out[:, 4:16] = scales_packed
    out[:, 16:48] = qh
    out[:, 48:176] = _pack_nibbles_q45(Ll)
    return out.reshape(nrow, -1)


def quantize_q4_K(x, qw=None):
    return _quantize_q45_K(x, qw, 15, False)


def quantize_q5_K(x, qw=None):
    return _quantize_q45_K(x, qw, 31, True)


def dequantize_q4_K(raw, n_per_row):
    nrow = raw.shape[0]
    b = raw.reshape(-1, 144)
    NB = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(F32).reshape(NB)
    dmin = b[:, 2:4].copy().view(np.float16).astype(F32).reshape(NB)
    sc, mn = _unpack_scales_k4(b[:, 4:16])
    L = _unpack_nibbles_q45(b[:, 16:144]).astype(F32)
    dl = d[:, None] * sc.astype(F32)
    ml = dmin[:, None] * mn.astype(F32)
    y = np.repeat(dl, 32, 1) * L - np.repeat(ml, 32, 1)
    return y.reshape(nrow, n_per_row)


def dequantize_q5_K(raw, n_per_row):
    nrow = raw.shape[0]
    b = raw.reshape(-1, 176)
    NB = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(F32).reshape(NB)
    dmin = b[:, 2:4].copy().view(np.float16).astype(F32).reshape(NB)
    sc, mn = _unpack_scales_k4(b[:, 4:16])
    qh = b[:, 16:48]
    Ll = _unpack_nibbles_q45(b[:, 48:176]).astype(np.int32)
    hi = np.zeros((NB, 256), np.int32)
    for c in range(4):
        hi[:, c * 64:c * 64 + 32] = (qh >> (2 * c)) & 1
        hi[:, c * 64 + 32:c * 64 + 64] = (qh >> (2 * c + 1)) & 1
    L = (Ll + 16 * hi).astype(F32)
    dl = d[:, None] * sc.astype(F32)
    ml = dmin[:, None] * mn.astype(F32)
    y = np.repeat(dl, 32, 1) * L - np.repeat(ml, 32, 1)
    return y.reshape(nrow, n_per_row)


# ---------------------------------------------------------------- q6_K

def quantize_q6_K(x, qw=None):
    nrow, npr = x.shape
    xb = x.reshape(-1, QK_K).astype(F32)
    NB = xb.shape[0]
    xs = xb.reshape(NB, 16, 16)

    if qw is None:
        scale, L = make_qx_quants(xs.reshape(-1, 16), None, 32)
    else:
        qwb = np.broadcast_to(np.asarray(qw, F32), (nrow, npr)).reshape(-1, QK_K)
        scale, L = make_qx_quants(xs.reshape(-1, 16), qwb.reshape(-1, 16), 32)
    scales = scale.reshape(NB, 16)
    L = L.reshape(NB, 256).astype(np.int32)  # 0..63 (offset +32)

    aidx = np.abs(scales).argmax(1)
    max_abs = np.abs(scales)[np.arange(NB), aidx]
    max_scale = scales[np.arange(NB), aidx]
    dead = max_abs < GROUP_MAX_EPS
    mxs = np.where(dead, np.float32(1), max_scale)
    iscale = np.float32(-128) / mxs
    d = np.float32(1) / iscale
    sc_i = np.minimum(nearest_int(iscale[:, None] * scales), 127).astype(np.int8)
    d16 = _f16bits(np.where(dead, np.float32(0), d))
    dsub = d16.astype(F32)[:, None] * sc_i.astype(F32)
    dsub_e = np.repeat(dsub, 16, 1)
    nz = (dsub_e != 0) & ~dead[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.clip(nearest_int(xb / np.where(nz, dsub_e, 1)), -32, 31) + 32
    L = np.where(nz, lq, L)
    L = np.where(dead[:, None], 0, L).astype(np.uint8)
    sc_i = np.where(dead[:, None], np.int8(0), sc_i)

    ql = np.zeros((NB, 128), np.uint8)
    qh = np.zeros((NB, 64), np.uint8)
    for half in range(2):  # 128-chunks
        base = half * 128
        seg = L[:, base:base + 128].reshape(NB, 4, 32)  # l, l+32, l+64, l+96
        ql[:, half * 64:half * 64 + 32] = (seg[:, 0] & 0xF) | ((seg[:, 2] & 0xF) << 4)
        ql[:, half * 64 + 32:half * 64 + 64] = (seg[:, 1] & 0xF) | ((seg[:, 3] & 0xF) << 4)
        qh[:, half * 32:half * 32 + 32] = ((seg[:, 0] >> 4) | ((seg[:, 1] >> 4) << 2) |
                                           ((seg[:, 2] >> 4) << 4) | ((seg[:, 3] >> 4) << 6))
    out = np.zeros((NB, 210), np.uint8)
    out[:, 0:128] = ql
    out[:, 128:192] = qh
    out[:, 192:208] = sc_i.view(np.uint8)
    out[:, 208:210] = d16.view(np.uint8).reshape(NB, 2)
    return out.reshape(nrow, -1)


def dequantize_q6_K(raw, n_per_row):
    nrow = raw.shape[0]
    b = raw.reshape(-1, 210)
    NB = b.shape[0]
    d = b[:, 208:210].copy().view(np.float16).astype(F32).reshape(NB)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.int32)  # [NB,16]
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    L = np.zeros((NB, 256), np.int32)
    for half in range(2):
        qlh = ql[:, half * 64:(half + 1) * 64]
        qhh = qh[:, half * 32:(half + 1) * 32]
        base = half * 128
        L[:, base:base + 32] = (qlh[:, 0:32] & 0xF) | (((qhh >> 0) & 3) << 4)
        L[:, base + 32:base + 64] = (qlh[:, 32:64] & 0xF) | (((qhh >> 2) & 3) << 4)
        L[:, base + 64:base + 96] = (qlh[:, 0:32] >> 4) | (((qhh >> 4) & 3) << 4)
        L[:, base + 96:base + 128] = (qlh[:, 32:64] >> 4) | (((qhh >> 6) & 3) << 4)
    q = (L - 32).astype(F32)
    dl = d[:, None] * np.repeat(sc, 16, 1).astype(F32)
    y = dl * q
    return y.reshape(nrow, n_per_row)


# ---------------------------------------------------------------- q8_0

def quantize_q8_0(x, qw=None):
    nrow, npr = x.shape
    xb = x.reshape(-1, 32).astype(F32)
    NB = xb.shape[0]
    amax = np.abs(xb).max(1)
    d = amax / np.float32(127)
    id_ = np.where(d != 0, np.float32(1) / d, np.float32(0))
    x0 = xb * id_[:, None]
    q = np.sign(x0) * np.floor(np.abs(x0) + np.float32(0.5))  # roundf: half away from zero
    out = np.zeros((NB, 34), np.uint8)
    out[:, 0:2] = _f16bits(d).view(np.uint8).reshape(NB, 2)
    out[:, 2:34] = q.astype(np.int8).view(np.uint8)
    return out.reshape(nrow, -1)


def dequantize_q8_0(raw, n_per_row):
    nrow = raw.shape[0]
    b = raw.reshape(-1, 34)
    NB = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(F32).reshape(NB)
    q = b[:, 2:34].copy().view(np.int8).astype(F32)
    return (d[:, None] * q).reshape(nrow, n_per_row)


# ---------------------------------------------------------------- dispatch

QUANTIZE = {"Q2_K": quantize_q2_K, "Q3_K": quantize_q3_K, "Q4_K": quantize_q4_K,
            "Q5_K": quantize_q5_K, "Q6_K": quantize_q6_K, "Q8_0": quantize_q8_0}
DEQUANTIZE = {"Q2_K": dequantize_q2_K, "Q3_K": dequantize_q3_K, "Q4_K": dequantize_q4_K,
              "Q5_K": dequantize_q5_K, "Q6_K": dequantize_q6_K, "Q8_0": dequantize_q8_0}


def roundtrip(x, ttype, qw=None):
    """quantize then dequantize; returns reconstruction."""
    raw = QUANTIZE[ttype](x, qw)
    return DEQUANTIZE[ttype](raw, x.shape[1])
