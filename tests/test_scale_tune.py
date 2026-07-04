"""scale_tune exactness: the linear d/dmin decomposition reproduces DEQUANTIZE bit-for-bit,
unchanged write_back is byte-identical, perturbed d round-trips through fp16 with codes
fixed, and the torch reconstruct matches with gradients flowing to d."""
import numpy as np
import pytest

from coldpress import kquant as kq
from coldpress import scale_tune as st
from conftest import TYPES, heavy_tailed


@pytest.mark.parametrize("tt", TYPES)
@pytest.mark.parametrize("qw_mode", ["ref", "imatrix"])
def test_extract_linear_exact(tt, qw_mode):
    nrow, n = 8, 512
    x = heavy_tailed(nrow, n, seed=hash(tt) & 0xFFFF)
    qw = None if qw_mode == "ref" else np.abs(heavy_tailed(1, n, seed=3))[0] + 1e-3
    raw = kq.QUANTIZE[tt](x, qw)
    ref = kq.DEQUANTIZE[tt](raw, n)
    lin = st.extract_linear(raw, tt, n)
    recon = st.reconstruct_np(lin)
    assert np.array_equal(recon, ref)


@pytest.mark.parametrize("tt", TYPES)
def test_unchanged_writeback_byte_identical(tt):
    nrow, n = 8, 512
    x = heavy_tailed(nrow, n, seed=99)
    raw = kq.QUANTIZE[tt](x, None)
    lin = st.extract_linear(raw, tt, n)
    raw_wb = st.write_back(raw, tt, lin["d"], lin["dmin"])
    assert np.array_equal(raw_wb, raw)


@pytest.mark.parametrize("tt", TYPES)
def test_perturb_roundtrips_fp16_codes_fixed(tt):
    nrow, n = 8, 512
    x = heavy_tailed(nrow, n, seed=5)
    raw = kq.QUANTIZE[tt](x, None)
    lin = st.extract_linear(raw, tt, n)
    d_pert = (lin["d"] * np.float32(1.01)).astype(np.float32)
    dmin_pert = (lin["dmin"] * np.float32(1.02)).astype(np.float32) if lin["dmin"] is not None else None
    raw_p = st.write_back(raw, tt, d_pert, dmin_pert)
    lin_p = st.extract_linear(raw_p, tt, n)
    assert np.array_equal(lin_p["d"], kq.fp16rt(d_pert))
    if dmin_pert is not None:
        assert np.array_equal(lin_p["dmin"], kq.fp16rt(dmin_pert))
    assert np.array_equal(lin_p["A"], lin["A"])
    if lin["B"] is not None:
        assert np.array_equal(lin_p["B"], lin["B"])
    assert not np.array_equal(lin_p["d"], lin["d"])  # change actually registered


@pytest.mark.parametrize("tt", TYPES)
def test_torch_reconstruct_and_grad(tt):
    import torch
    nrow, n = 8, 512
    x = heavy_tailed(nrow, n, seed=11)
    raw = kq.QUANTIZE[tt](x, None)
    ref = kq.DEQUANTIZE[tt](raw, n)
    lin = st.extract_linear(raw, tt, n)
    entry = {
        "d": torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(lin["d"], np.float32))),
        "dmin": (torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(lin["dmin"], np.float32)))
                 if lin["dmin"] is not None else None),
        "A": torch.from_numpy(np.ascontiguousarray(lin["A"]).astype(np.int16)),
        "B": (torch.from_numpy(np.ascontiguousarray(lin["B"]).astype(np.int16))
              if lin["B"] is not None else None),
        "ttype": tt, "shape": lin["shape"],
    }
    W = st.reconstruct(entry).detach().numpy()
    assert np.array_equal(W, ref)
    entry["d"].grad = None
    (st.reconstruct(entry) ** 2).sum().backward()
    assert entry["d"].grad is not None and torch.isfinite(entry["d"].grad).all()


def test_build_torch_params_on_synthetic_gguf(tmp_path):
    import gguf
    import torch
    nrow, n, tt = 8, 512, "Q4_K"
    x = heavy_tailed(nrow, n, seed=999)
    raw = kq.QUANTIZE[tt](x, None)
    ref = kq.DEQUANTIZE[tt](raw, n)
    path = str(tmp_path / "tiny.gguf")
    w = gguf.GGUFWriter(path, "tiny")
    w.add_tensor("blk.0.weight", np.ascontiguousarray(raw, np.uint8),
                 raw_shape=raw.shape, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    params = st.build_torch_params(path)
    entry = params["blk.0.weight"]
    W = st.reconstruct(entry).detach().numpy()
    assert entry["shape"] == (nrow, n)
    assert np.array_equal(W, ref)
