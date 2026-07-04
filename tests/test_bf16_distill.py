"""bf16 distill skeleton numerics (SCALE blocker #1):

  * reconstruct() computes in float32 and casts to the skeleton dtype as the LAST step, so
    reconstruct-in-f32-then-cast is bit-identical to casting the f32 result, and within bf16
    eps of the pure-f32 reconstruction.
  * scale gradients flow (finite, nonzero) from a bf16 forward back to the float32 d/dmin
    params through the cast -- exactly the E3B functional_call skeleton at bf16 on CPU.
"""
import numpy as np
import pytest

from coldpress import kquant as kq
from coldpress import scale_tune as st
from conftest import TYPES, heavy_tailed


def _entry(tt, nrow=8, n=512, seed=1):
    import torch
    x = heavy_tailed(nrow, n, seed=seed)
    raw = kq.QUANTIZE[tt](x, None)
    lin = st.extract_linear(raw, tt, n)
    return {
        "d": torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(lin["d"], np.float32))),
        "dmin": (torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(lin["dmin"], np.float32)))
                 if lin["dmin"] is not None else None),
        "A": torch.from_numpy(np.ascontiguousarray(lin["A"]).astype(np.int16)),
        "B": (torch.from_numpy(np.ascontiguousarray(lin["B"]).astype(np.int16))
              if lin["B"] is not None else None),
        "ttype": tt, "shape": lin["shape"],
    }


@pytest.mark.parametrize("tt", TYPES)
def test_reconstruct_f32_then_cast_within_bf16_eps(tt):
    import torch
    e = _entry(tt)
    W_f32 = st.reconstruct(e)                              # float32 (bit-exact vs DEQUANTIZE)
    W_bf16 = st.reconstruct(e, dtype=torch.bfloat16)       # cast at the end
    assert W_f32.dtype == torch.float32 and W_bf16.dtype == torch.bfloat16
    # reconstruct-in-f32-then-cast == cast-the-f32-result, bit for bit
    assert torch.equal(W_bf16, W_f32.to(torch.bfloat16))
    # ...and that is within bf16 eps of the pure-f32 reconstruction
    eps = torch.finfo(torch.bfloat16).eps
    tol = eps * float(W_f32.abs().max()) + 1e-6
    assert float((W_bf16.float() - W_f32).abs().max()) <= tol


@pytest.mark.parametrize("tt", ["Q2_K", "Q4_K", "Q6_K"])
def test_scale_grads_flow_through_bf16_skeleton_cpu(tt):
    """A tiny bf16 skeleton on CPU whose first weight is reconstructed from f32 scale params:
    a bf16 forward + backward must leave finite, nonzero grads on the float32 d (and dmin)."""
    import torch
    import torch.nn as nn
    from torch.func import functional_call

    e = _entry(tt, nrow=8, n=512, seed=2)
    n_out, n_in = e["shape"]
    # a 2-linear bf16 skeleton (no saturating nonlinearity -- Q6_K weights reach ~10^3, which
    # would push a tanh into saturation and kill the gradient signal, defeating the point of
    # the check). Small input keeps the bf16 forward well-scaled.
    skeleton = nn.Sequential(
        nn.Linear(n_in, n_out, bias=False),
        nn.Linear(n_out, 4, bias=False),
    ).to(torch.bfloat16)
    for p in skeleton.parameters():
        p.requires_grad_(False)
    inp = (torch.randn(3, n_in) * 0.1).to(torch.bfloat16)

    W = st.reconstruct(e, dtype=torch.bfloat16)   # [n_out, n_in] bf16, grad to f32 d/dmin
    assert W.dtype == torch.bfloat16
    params = {"0.weight": W}
    out = functional_call(skeleton, params, (inp,))
    loss = out.float().pow(2).sum()
    loss.backward()

    assert e["d"].grad is not None
    assert torch.isfinite(e["d"].grad).all()
    assert float(e["d"].grad.abs().sum()) > 0.0, "no gradient reached the f32 scale param"
    if e["dmin"] is not None:
        assert e["dmin"].grad is not None and torch.isfinite(e["dmin"].grad).all()
