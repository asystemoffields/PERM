#!/usr/bin/env python3
"""Reserve-lever: post-quant distillation of the fp16 superblock scales (d, dmin).

The quantized integer codes stay FIXED; only the continuous per-superblock fp16
scale fields (d, and dmin where the type has one) are tuned. Dequantization is
LINEAR in (d, dmin) given the fixed codes, so a quantized tensor can be written

    W_hat[r, c] = d[r, sb(c)] * A[r, c]  -  dmin[r, sb(c)] * B[r, c]

where A, B are constant integer-product tensors read out of the container bytes:

    A[r, c] = subscale_int[r, block(c)] * q_signed[r, c]      (the integer d multiplies)
    B[r, c] = minside_int[r, block(c)]                        (the integer dmin multiplies)

sb(c)    = c // 256                (superblock index; superblocks are 256 elements)
block(c) = c // SUB_W[ttype]       (sub-block index within the row)

Per-type semantics (from /data/coldpress/results/kquant-reference.md, matched byte-for-byte
against kquant.DEQUANTIZE):
  Q2_K : 16 subblocks/superblock, SUB_W=16. sc=scales[j]. A=(sc&0xF)*q, q in 0..3.  B=(sc>>4).
  Q3_K : 16 subblocks/superblock, SUB_W=16. A=(sc-32)*q_signed, q in -4..3.          B=None (no dmin).
  Q4_K :  8 subblocks/superblock, SUB_W=32. A=sc6*q, q in 0..15.                      B=m6.
  Q5_K :  8 subblocks/superblock, SUB_W=32. A=sc6*q, q in 0..31.                      B=m6.
  Q6_K : 16 subblocks/superblock, SUB_W=16. A=int8_sc*(q-32), q-32 in -32..31.        B=None (no dmin).

EXACTNESS: with d/dmin at their fp16-round-tripped stored values, the linear form above
computed in float32 equals kquant.DEQUANTIZE bit-for-bit. This holds despite the
associativity difference vs the decoder's ((d*sc)*q) because every intermediate integer
product fits the fp32 24-bit mantissa (fp16 d contributes <=11 significant bits; the largest
|A| is Q6_K's 128*32=4096 which multiplied by an 11-bit d needs <=23 bits). Verified in tests.

STORAGE: A ranges up to +-4096 (Q6_K) / 3969 odd (Q5_K), which are NOT all exact in float16
(integers are exact only up to 2048, then even up to 4096). So A/B are stored as int16 torch
tensors (all magnitudes fit int16 exactly) and cast to float32 at reconstruct time -- keeping
memory sane while staying bit-exact.
"""
import numpy as np

from . import kquant as kq

F32 = np.float32

# bytes per superblock, sub-block width, and byte offsets of the fp16 d / dmin fields.
# dmin_off is None for types with no min side (Q3_K, Q6_K).
_LAYOUT = {
    "Q2_K": dict(bpsb=84,  sub_w=16, d_off=80,  dmin_off=82),
    "Q3_K": dict(bpsb=110, sub_w=16, d_off=108, dmin_off=None),
    "Q4_K": dict(bpsb=144, sub_w=32, d_off=0,   dmin_off=2),
    "Q5_K": dict(bpsb=176, sub_w=32, d_off=0,   dmin_off=2),
    "Q6_K": dict(bpsb=210, sub_w=16, d_off=208, dmin_off=None),
}
SUPPORTED = tuple(_LAYOUT)


def _f16_field(b, off):
    """Read a little-endian fp16 field at byte offset `off` from block rows b -> f32 [NB]."""
    return b[:, off:off + 2].copy().view(np.float16).astype(F32).reshape(b.shape[0])


def _expand_sub(sub_int, sub_w):
    """[NB, n_sub] integer sub-block values -> [NB, 256] by repeating each sub_w times."""
    return np.repeat(sub_int, sub_w, axis=1)


# ---------------------------------------------------------------- extract_linear

def extract_linear(raw_bytes_2d, ttype, n_per_row):
    """Decompose a k-quant tensor's dequant into W_hat = d (x) A - dmin (x) B.

    raw_bytes_2d : uint8 [nrow, row_bytes] container bytes (row_bytes = n_per_row//256 * bpsb).
    ttype        : one of Q2_K/Q3_K/Q4_K/Q5_K/Q6_K.
    n_per_row    : logical element count per row (multiple of 256).

    Returns dict:
      d    : f32 [nrow, n_sb]        superblock d fields (fp16-round-tripped values)
      dmin : f32 [nrow, n_sb] | None superblock dmin fields (None for Q3_K/Q6_K)
      A    : f32 [nrow, n_per_row]   subscale_int * q_signed  (what d multiplies)
      B    : f32 [nrow, n_per_row] | None   min-side integer  (what dmin multiplies)
      ttype, shape=(nrow, n_per_row)

    Guarantee: d_exp * A - dmin_exp * B (float32) == kquant.DEQUANTIZE[ttype](raw, n_per_row)
    exactly (np.array_equal), where *_exp repeats the per-superblock scalar over 256 elements.
    """
    if ttype not in _LAYOUT:
        raise ValueError(f"unsupported ttype {ttype!r}; supported: {SUPPORTED}")
    lay = _LAYOUT[ttype]
    bpsb, sub_w = lay["bpsb"], lay["sub_w"]
    raw = np.ascontiguousarray(raw_bytes_2d, dtype=np.uint8)
    nrow = raw.shape[0]
    n_sb = n_per_row // 256
    b = raw.reshape(-1, bpsb)          # [NB, bpsb], NB = nrow * n_sb
    NB = b.shape[0]

    d = _f16_field(b, lay["d_off"])    # [NB]
    dmin = None
    if lay["dmin_off"] is not None:
        dmin = _f16_field(b, lay["dmin_off"])

    if ttype == "Q2_K":
        sc = b[:, 0:16].astype(np.int32)          # [NB,16]
        subscale = sc & 0xF                        # 0..15
        minint = sc >> 4                           # 0..15
        q = kq._unpack_qs_2bit(b[:, 16:80]).astype(np.int32)   # [NB,256] 0..3
        A = _expand_sub(subscale, sub_w).astype(F32) * q.astype(F32)
        B = _expand_sub(minint, sub_w).astype(F32)

    elif ttype == "Q3_K":
        sc = (kq._unpack_scales_q3(b[:, 96:108]) - 32).astype(np.int32)   # [NB,16] signed
        lo = kq._unpack_qs_2bit(b[:, 32:96]).astype(np.int32)            # 0..3
        hmask = b[:, 0:32]
        hi = np.zeros((NB, 256), np.int32)
        for j in range(8):
            hi[:, j * 32:(j + 1) * 32] = (hmask >> j) & 1
        q = lo - np.where(hi == 0, 4, 0)          # signed -4..3 (cleared bit subtracts 4)
        A = _expand_sub(sc, sub_w).astype(F32) * q.astype(F32)
        B = None

    elif ttype in ("Q4_K", "Q5_K"):
        # both: d@0 dmin@2 scales@4:16; Q4_K qs@16:144, Q5_K qh@16:48 qs@48:176
        sc_i, mn_i = kq._unpack_scales_k4(b[:, 4:16])   # [NB,8] each 0..63
        if ttype == "Q4_K":
            q = kq._unpack_nibbles_q45(b[:, 16:144]).astype(np.int32)     # 0..15
        else:
            Ll = kq._unpack_nibbles_q45(b[:, 48:176]).astype(np.int32)    # 0..15
            qh = b[:, 16:48]
            hi = np.zeros((NB, 256), np.int32)
            for c in range(4):
                hi[:, c * 64:c * 64 + 32] = (qh >> (2 * c)) & 1
                hi[:, c * 64 + 32:c * 64 + 64] = (qh >> (2 * c + 1)) & 1
            q = Ll + 16 * hi                                             # 0..31
        A = _expand_sub(sc_i.astype(np.int32), sub_w).astype(F32) * q.astype(F32)
        B = _expand_sub(mn_i.astype(np.int32), sub_w).astype(F32)

    elif ttype == "Q6_K":
        sc = b[:, 192:208].copy().view(np.int8).astype(np.int32)   # [NB,16] signed int8
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
        q = L - 32                                # signed -32..31
        A = _expand_sub(sc, sub_w).astype(F32) * q.astype(F32)
        B = None

    else:  # pragma: no cover
        raise ValueError(ttype)

    d2 = d.reshape(nrow, n_sb)
    dmin2 = dmin.reshape(nrow, n_sb) if dmin is not None else None
    A2 = A.reshape(nrow, n_per_row)
    B2 = B.reshape(nrow, n_per_row) if B is not None else None
    return {"d": d2, "dmin": dmin2, "A": A2, "B": B2,
            "ttype": ttype, "shape": (nrow, n_per_row)}


def reconstruct_np(lin):
    """numpy float32 reconstruction from an extract_linear dict (matches DEQUANTIZE exactly)."""
    d, A = lin["d"], lin["A"]
    nrow, n_per_row = lin["shape"]
    d_exp = np.repeat(d, 256, axis=1)
    W = d_exp * A
    if lin["dmin"] is not None:
        dmin_exp = np.repeat(lin["dmin"], 256, axis=1)
        W = W - dmin_exp * lin["B"]
    return W.astype(F32)


# ---------------------------------------------------------------- write_back

def write_back(raw_bytes_2d, ttype, d_new_f32, dmin_new_f32=None):
    """Return new container bytes with ONLY the fp16 d/dmin fields replaced.

    d_new_f32    : f32 [nrow, n_sb] new superblock d values (cast f32 -> f16 on write).
    dmin_new_f32 : f32 [nrow, n_sb] new dmin values, or None. Ignored for Q3_K/Q6_K.
    Every other byte is copied unchanged.
    """
    if ttype not in _LAYOUT:
        raise ValueError(f"unsupported ttype {ttype!r}")
    lay = _LAYOUT[ttype]
    bpsb = lay["bpsb"]
    raw = np.ascontiguousarray(raw_bytes_2d, dtype=np.uint8)
    nrow = raw.shape[0]
    b = raw.reshape(-1, bpsb).copy()
    NB = b.shape[0]

    d16 = np.asarray(d_new_f32, F32).reshape(NB).astype(np.float16)
    b[:, lay["d_off"]:lay["d_off"] + 2] = d16.view(np.uint8).reshape(NB, 2)

    if lay["dmin_off"] is not None:
        if dmin_new_f32 is None:
            raise ValueError(f"{ttype} has a dmin field; dmin_new_f32 required")
        dm16 = np.asarray(dmin_new_f32, F32).reshape(NB).astype(np.float16)
        b[:, lay["dmin_off"]:lay["dmin_off"] + 2] = dm16.view(np.uint8).reshape(NB, 2)

    return b.reshape(nrow, -1)


# ---------------------------------------------------------------- torch params

def build_torch_params(gguf_path):
    """Read every quantized (Q2..Q6_K) tensor from a GGUF and return trainable scale params.

    Returns {name: entry} where entry = {
        "d": torch.nn.Parameter f32 [nrow, n_sb],
        "dmin": Parameter f32 [nrow, n_sb] or None,
        "A": torch int16 const [nrow, n_per_row],
        "B": torch int16 const [nrow, n_per_row] or None,
        "ttype": str, "shape": (nrow, n_per_row),
    }
    A/B are int16 (exact for all k-quant integer products) and cast to f32 at reconstruct time.
    """
    import torch
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    out = {}
    for t in reader.tensors:
        tt = t.tensor_type.name
        if tt not in _LAYOUT:
            continue
        n_per_row = int(t.shape[0])           # GGUF dims[0] is the innermost (row) length
        nrow = int(t.n_elements) // n_per_row
        raw = np.ascontiguousarray(t.data).reshape(nrow, -1).astype(np.uint8)
        lin = extract_linear(raw, tt, n_per_row)
        entry = {
            "d": torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(lin["d"], F32))),
            "dmin": (torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(lin["dmin"], F32)))
                     if lin["dmin"] is not None else None),
            "A": torch.from_numpy(np.ascontiguousarray(lin["A"]).astype(np.int16)),
            "B": (torch.from_numpy(np.ascontiguousarray(lin["B"]).astype(np.int16))
                  if lin["B"] is not None else None),
            "ttype": tt,
            "shape": lin["shape"],
        }
        out[t.name] = entry
    return out


def reconstruct(entry):
    """torch f32 weight [nrow, n_per_row] = d_expanded * A - dmin_expanded * B.

    d/dmin are per-superblock scalars [nrow, n_sb] expanded to per-element via
    repeat_interleave(256) (superblocks are 256 elements). A/B are int16 consts cast to f32.
    Differentiable wrt d (and dmin); A/B carry no grad.
    """
    import torch
    d = entry["d"]
    A = entry["A"].to(torch.float32)
    n_sb = d.shape[1]
    W = d.repeat_interleave(256, dim=1) * A
    if entry["dmin"] is not None:
        dmin = entry["dmin"]
        B = entry["B"].to(torch.float32)
        W = W - dmin.repeat_interleave(256, dim=1) * B
    return W
