"""decoders-vs-gguf-py: our kquant decoders are bit-exact vs the pip gguf reference.

Pure pip deps (numpy + gguf). We encode with our faithful encoder, then check that BOTH
our decoder and gguf.dequantize reconstruct identically -- pinning the container layout to
the ecosystem's own decoder."""
import numpy as np
import pytest
import gguf

from coldpress import kquant as kq
from conftest import TYPES, heavy_tailed


@pytest.mark.parametrize("tt", TYPES + ["Q8_0"])
@pytest.mark.parametrize("qw_mode", ["ref", "imatrix"])
def test_decoder_matches_gguf_dequantize(tt, qw_mode):
    nrow, n = 6, 512
    x = heavy_tailed(nrow, n, seed=hash((tt, qw_mode)) & 0xFFFF)
    qw = None
    if qw_mode == "imatrix":
        qw = np.abs(heavy_tailed(1, n, seed=7))[0] + 1e-3
    raw = kq.QUANTIZE[tt](x, qw)          # [nrow, row_bytes]
    ours = kq.DEQUANTIZE[tt](raw, n)
    qtype = getattr(gguf.GGMLQuantizationType, tt)
    ref = gguf.dequantize(np.ascontiguousarray(raw), qtype).reshape(nrow, n)
    assert np.array_equal(ours.astype(np.float32), ref.astype(np.float32)), \
        f"{tt}/{qw_mode}: max abs diff {np.abs(ours - ref).max()}"
