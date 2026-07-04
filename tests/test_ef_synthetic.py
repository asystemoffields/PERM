"""EF-beats-weighted-RTN synthetic: with a correlated input Hessian, GPTQ error feedback
lowers the Hessian-weighted reconstruction error trace((W-W_hat) H (W-W_hat)^T) below
plain weighted-RTN into the same container. (EF trades DIAGONAL error for correlated
compensation, so this must be measured on the FULL quadratic form, not diag(H).)"""
import numpy as np
import pytest

from coldpress import kquant as kq
from coldpress import ef


def _correlated_hessian(n, T, seed):
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((n, n)) / np.sqrt(n)
    mix += 1.5 * np.eye(n)
    Z = rng.standard_normal((T, n))
    X = Z @ mix.T
    H = (X.T @ X).astype(np.float64)
    return H


def _h_weighted_err(W, What, H):
    D = (What - W).astype(np.float64)
    return float(np.einsum("ri,ij,rj->", D, H, D))


_SEED = {"Q2_K": 101, "Q3_K": 202, "Q4_K": 303, "Q5_K": 404, "Q6_K": 505}


@pytest.mark.parametrize("tt", ["Q2_K", "Q3_K", "Q4_K"])
def test_ef_beats_weighted_rtn(tt):
    n, nrow, T = 512, 24, 4096
    # deterministic per-type seed (Python's hash() is salted per process -> was flaky)
    rng = np.random.default_rng(_SEED[tt])
    W = (rng.standard_normal((nrow, n)) ** 3).astype(np.float32)
    H = _correlated_hessian(n, T, seed=0)
    qw = (np.diag(H) / T).astype(np.float32)   # imatrix means for the weighted-RTN baseline

    rtn = kq.roundtrip(W, tt, qw)
    _raw, ef_recon = ef.ef_encode(W, tt, qw, H, n_iter=2, act_order=True)

    e_rtn = _h_weighted_err(W, rtn, H)
    e_ef = _h_weighted_err(W, ef_recon, H)
    assert e_ef < e_rtn, f"{tt}: EF H-weighted err {e_ef:.4g} !< RTN {e_rtn:.4g}"
    # meaningful improvement, not numerical noise
    assert e_ef < 0.97 * e_rtn, f"{tt}: EF improvement only {(1 - e_ef / e_rtn) * 100:.2f}%"


@pytest.mark.parametrize("tt", ["Q2_K", "Q3_K", "Q4_K"])
def test_act_order_beats_storage_order_when_importance_last(tt):
    """The measured failure mode (hard rule 1): when the most important channels sit LAST in
    storage order -- exactly what magnitude-sorted PERM produces -- a storage-order GPTQ
    sweep dumps its accumulated error into them. act_order (process by descending diag(H),
    decoupled from the committed storage-order grid) avoids that. Here we build that
    importance-last ordering and require act_order's H-weighted error to be strictly lower."""
    n, nrow, T = 512, 24, 8192
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((nrow, n)) ** 3).astype(np.float32)
    var = 1.0 + 8.0 * np.arange(n) / n            # importance increases with column index
    Z = rng.standard_normal((T, n)) * np.sqrt(var)
    Z += 0.4 * Z[:, [0]]                            # correlation across columns
    H = (Z.T @ Z).astype(np.float64)
    qw = (np.diag(H) / T).astype(np.float32)
    _r1, rec_act = ef.ef_encode(W, tt, qw, H, n_iter=2, act_order=True)
    _r2, rec_sto = ef.ef_encode(W, tt, qw, H, n_iter=2, act_order=False)
    e_act = _h_weighted_err(W, rec_act, H)
    e_sto = _h_weighted_err(W, rec_sto, H)
    assert e_act < e_sto, f"{tt}: act_order {e_act:.4g} !< storage_order {e_sto:.4g}"
