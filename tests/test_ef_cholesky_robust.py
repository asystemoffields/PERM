"""cholesky_inv_upper must survive a numerically borderline Hessian.

The GPTQ inverse-Hessian Cholesky (ef.cholesky_inv_upper) used a single fixed damping and a
hard np.linalg.cholesky. On a real ffn tensor -- ill-conditioned X^T X, off-diagonals stored
float16 (SCALE blocker #2), permuted column order -- inv(H) picked up a tiny negative
eigenvalue and cholesky raised "Matrix is not positive definite", killing a ~5.6 h pkgval run
at the final encode. The fix escalates the diagonal load until the factorization succeeds while
leaving the well-conditioned path bit-identical (see cholesky_inv_upper docstring)."""
import numpy as np

from coldpress import hessians as hess
from coldpress.ef import cholesky_inv_upper


def _spectrum_matrix(n, eig, seed=0):
    """Symmetric matrix with an exactly-specified eigenspectrum in a random orthonormal basis."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    M = (Q * eig) @ Q.T
    return (M + M.T) / 2


def _naive_fixed_damp_fails(H, damp=0.01):
    """Reproduce the OLD code path exactly; True if it raises LinAlgError."""
    n = H.shape[0]
    Hd = H.astype(np.float64).copy()
    Hd[np.diag_indices(n)] += damp * float(np.mean(np.diag(Hd))) + 1e-8
    try:
        Hinv = np.linalg.inv(Hd)
        np.linalg.cholesky((Hinv + Hinv.T) / 2)
        return False
    except np.linalg.LinAlgError:
        return True


def test_escalation_recovers_non_pd_input():
    """A matrix with one clearly-negative eigenvalue (|lambda| > damp*mean_diag) reliably breaks
    the fixed-damp path; escalation must recover a finite factor. Deterministic (exact spectrum),
    so it is a genuine regression guard rather than a numerically flaky one."""
    n = 256
    eig = np.linspace(1.0, 100.0, n)   # PSD-like bulk, mean_diag ~ 50
    eig[0] = -2.0                       # perturbation-induced negative eigenvalue
    H = _spectrum_matrix(n, eig, seed=1)
    assert _naive_fixed_damp_fails(H), "test input no longer reproduces the failure mode"
    U = cholesky_inv_upper(H, damp=0.01)
    assert U.shape == (n, n)
    assert np.all(np.isfinite(U))
    assert np.allclose(U, np.triu(U)), "U must be upper-triangular"


def test_wellconditioned_path_is_byte_identical():
    """On a comfortably PD Hessian the escalation loop's first attempt must reproduce the
    historical fixed-damp result bit-for-bit -- the frozen champion's encodes are unchanged."""
    n = 128
    rng = np.random.default_rng(5)
    M = rng.standard_normal((n, n))
    H = (M @ M.T).astype(np.float64) + n * np.eye(n)   # strongly PD
    # historical reference computation
    Href = H.copy()
    Href[np.diag_indices(n)] += 0.01 * float(np.mean(np.diag(Href))) + 1e-8
    Hinv = np.linalg.inv(Href)
    U_ref = np.ascontiguousarray(np.linalg.cholesky((Hinv + Hinv.T) / 2).T)
    U_new = cholesky_inv_upper(H, damp=0.01)
    assert np.array_equal(U_new, U_ref), "well-conditioned path drifted from fixed-damp result"


def test_fp16_stored_illconditioned_hessian_encodes(tmp_path):
    """Faithful path: an ill-conditioned X^T X pushed through the real fp16 save/load storage
    must not crash cholesky_inv_upper."""
    n = 512
    rng = np.random.default_rng(2)
    A = rng.standard_normal((n // 4, n))   # rank-deficient (fewer samples than dims)
    A[:4] *= 60.0                          # a few dominant directions -> large condition number
    H = (A.T @ A).astype(np.float32)
    hess.save_hessians({"blk.0.ffn_down.weight": {"H": H, "n": 4096}}, str(tmp_path))
    H_rt, _ = hess.load_hessian(str(tmp_path), "blk.0.ffn_down.weight")
    U = cholesky_inv_upper(H_rt, damp=0.01)
    assert U.shape == (n, n)
    assert np.all(np.isfinite(U))
