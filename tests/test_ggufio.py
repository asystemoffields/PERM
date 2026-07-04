"""G0: re-emitting a GGUF through our writer is byte-identical on tensor data + KV, and the
imatrix-means reader round-trips. Pure pip deps."""
import numpy as np
import gguf

from coldpress import kquant as kq
from coldpress import ggufio
from conftest import heavy_tailed


def _build_gguf(path):
    nrow, n = 8, 512
    raw = kq.QUANTIZE["Q2_K"](heavy_tailed(nrow, n, 1), None)
    w = gguf.GGUFWriter(path, "qwen3")
    w.add_uint32("qwen3.block_count", 2)
    w.add_string("general.name", "tiny")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    w.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_bool("some.bool", True)
    w.add_tensor("blk.0.weight", np.ascontiguousarray(raw, np.uint8),
                 raw_shape=raw.shape, raw_dtype=gguf.GGMLQuantizationType.Q2_K)
    w.add_tensor("norm.weight", heavy_tailed(1, 16, 2)[0])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_reemit_byte_identical(tmp_path):
    a = str(tmp_path / "a.gguf")
    b = str(tmp_path / "b.gguf")
    _build_gguf(a)
    ggufio.reemit(a, b)
    assert ggufio.compare(a, b)


def test_reemit_replace_quant_tensor(tmp_path):
    a = str(tmp_path / "a.gguf")
    b = str(tmp_path / "b.gguf")
    _build_gguf(a)
    # replace blk.0.weight with a different encoding of new data
    raw2 = kq.QUANTIZE["Q2_K"](heavy_tailed(8, 512, 42), None)
    ggufio.reemit(a, b, replace={"blk.0.weight": raw2.reshape(-1)})
    tm = ggufio.read_typemap(b)
    assert tm["blk.0.weight"]["type"] == "Q2_K"  # type/layout preserved
    assert not ggufio.compare(a, b)              # data changed


def test_typemap(tmp_path):
    a = str(tmp_path / "a.gguf")
    _build_gguf(a)
    tm = ggufio.read_typemap(a)
    assert tm["blk.0.weight"]["type"] == "Q2_K"
    assert tm["blk.0.weight"]["shape"][0] == 512  # ne[0] == n_per_row
    assert tm["norm.weight"]["type"] == "F32"
