"""fp16 Hessian storage (SCALE blocker #2): save_hessians stores H as float16 with an EXACT
float32 diagonal ("diag32"); load_hessian restores float64 H with the exact diagonal written
back over the fp16 one. Old float32-only npz still loads. And EF on the fp16-round-tripped H
of the existing synthetic case still beats weighted-RTN (reuses that test's harness)."""
import numpy as np
import pytest

from coldpress import hessians as hess
from coldpress import ef
from coldpress import kquant as kq
from test_ef_synthetic import _correlated_hessian, _h_weighted_err


def _sym_psd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M @ M.T).astype(np.float64) + n * np.eye(n)


def test_fp16_roundtrip_diag_exact_offdiag_within_eps(tmp_path):
    n = 128
    H = _sym_psd(n, seed=0)
    hess.save_hessians({"blk.0.attn_q.weight": {"H": H.astype(np.float32), "n": 777}},
                       str(tmp_path))
    Hr, ncount = hess.load_hessian(str(tmp_path), "blk.0.attn_q.weight")
    assert ncount == 777
    assert Hr.dtype == np.float64
    # diagonal is EXACT (f32 stored, f64 loaded)
    assert np.array_equal(np.diag(Hr), np.diag(H).astype(np.float32).astype(np.float64))
    # off-diagonal within fp16 relative eps. save_hessians stores float32(H)->float16, so the
    # reference goes through float32 first (matching the save path exactly; a direct f64->f16
    # can double-round differently).
    off = ~np.eye(n, dtype=bool)
    ref = H.astype(np.float32).astype(np.float16).astype(np.float64)
    assert np.array_equal(Hr[off], ref[off])
    eps = np.finfo(np.float16).eps
    denom = np.maximum(np.abs(H[off]), 1e-6)
    assert float(np.max(np.abs(Hr[off] - H[off]) / denom)) <= eps


def test_backward_compat_float32_npz(tmp_path):
    """An OLD-format npz (H f32, no diag32) still loads unchanged."""
    n = 16
    H = _sym_psd(n, seed=3).astype(np.float32)
    np.savez_compressed(str(tmp_path / "blk_0_attn_q_weight.npz"), H=H, n=42, name="x")
    Hr, ncount = hess.load_hessian(str(tmp_path), "blk.0.attn_q.weight")
    assert ncount == 42 and np.array_equal(Hr, H.astype(np.float64))


@pytest.mark.parametrize("tt", ["Q2_K", "Q3_K", "Q4_K"])
def test_ef_on_fp16_roundtripped_H_beats_stock(tt, tmp_path):
    """Persist the correlated synthetic Hessian through the fp16 save/load path, then run EF
    with the ROUND-TRIPPED H; its H-weighted error must still beat weighted-RTN (measured on
    the true H)."""
    n, nrow, T = 512, 24, 4096
    rng = np.random.default_rng(hash(tt) & 0xFFFF)
    W = (rng.standard_normal((nrow, n)) ** 3).astype(np.float32)
    H = _correlated_hessian(n, T, seed=0)
    qw = (np.diag(H) / T).astype(np.float32)

    hess.save_hessians({"blk.0.ffn_down.weight": {"H": H.astype(np.float32), "n": T}},
                       str(tmp_path))
    H_rt, _ = hess.load_hessian(str(tmp_path), "blk.0.ffn_down.weight")

    rtn = kq.roundtrip(W, tt, qw)
    _raw, ef_recon = ef.ef_encode(W, tt, qw, H_rt, n_iter=2, act_order=True)
    e_rtn = _h_weighted_err(W, rtn, H)
    e_ef = _h_weighted_err(W, ef_recon, H)
    assert e_ef < e_rtn, f"{tt}: EF (fp16-RT H) {e_ef:.4g} !< RTN {e_rtn:.4g}"
    assert e_ef < 0.97 * e_rtn, f"{tt}: EF improvement only {(1 - e_ef / e_rtn) * 100:.2f}%"
