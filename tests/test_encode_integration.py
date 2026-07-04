"""End-to-end EF driver on a synthetic artifact (no llama.cpp): build an f16 GGUF + a stock
quant target + per-tensor Hessians, run ef.encode_gguf, and assert the output passes the
byte-parity gate and decodes. Exercises the tier-1 path fully."""
import numpy as np
import gguf
from gguf import GGUFWriter, GGUFReader

from coldpress import kquant as kq
from coldpress import ef, gates
from coldpress import hessians as hess
from conftest import heavy_tailed

# (name, ttype, ne0, nrow)
SPEC = [
    ("token_embd.weight", "Q6_K", 256, 512),
    ("output.weight", "Q6_K", 256, 512),
    ("blk.0.attn_q.weight", "Q2_K", 256, 256),
    ("blk.0.ffn_down.weight", "Q2_K", 512, 256),
]


def _weights():
    return {name: heavy_tailed(nrow, ne0, seed=i)
            for i, (name, tt, ne0, nrow) in enumerate(SPEC)}


def _write_f16(path, W):
    w = GGUFWriter(path, "qwen3")
    w.add_uint32("qwen3.block_count", 1)
    for name, arr in W.items():
        w.add_tensor(name, arr.astype(np.float16))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def _write_stock(path, W):
    w = GGUFWriter(path, "qwen3")
    w.add_uint32("qwen3.block_count", 1)
    for name, tt, ne0, nrow in SPEC:
        raw = kq.QUANTIZE[tt](W[name].astype(np.float32), None)
        w.add_tensor(name, np.ascontiguousarray(raw, np.uint8),
                     raw_shape=raw.shape, raw_dtype=getattr(gguf.GGMLQuantizationType, tt))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def _write_hessians(hdir, W):
    for name, tt, ne0, nrow in SPEC:
        if name in ("token_embd.weight",):
            continue  # token_embd uses the attn_q diagonal proxy
        rng = np.random.default_rng(hash(name) & 0xFFFF)
        X = rng.standard_normal((2048, ne0))
        X += 0.3 * X[:, [0]]  # inject cross-column correlation
        H = (X.T @ X).astype(np.float32)
        hess.save_hessians({name: {"H": H, "n": 2048}}, hdir)


def test_encode_gguf_end_to_end(tmp_path):
    W = _weights()
    f16 = str(tmp_path / "f16.gguf")
    stock = str(tmp_path / "stock.gguf")
    out = str(tmp_path / "coldpress.gguf")
    hdir = str(tmp_path / "hess")
    _write_f16(f16, W)
    _write_stock(stock, W)
    _write_hessians(hdir, W)

    stats = ef.encode_gguf(f16, stock, hdir, out, imatrix_path=None, log=lambda *_: None)

    # byte-parity gate: same type map, <= bytes
    rep = gates.byte_parity(out, stock)
    assert rep["passed"] and rep["total_final"] <= rep["total_stock"]

    # EF ran on the tensors with Hessians and actually changed their encoding
    assert stats["blk.0.attn_q.weight"]["mode"] == "EF"
    assert stats["token_embd.weight"]["mode"].startswith("wRTN")
    a = {t.name: np.asarray(t.data).tobytes() for t in GGUFReader(stock).tensors}
    b = {t.name: np.asarray(t.data).tobytes() for t in GGUFReader(out).tensors}
    assert a["blk.0.attn_q.weight"] != b["blk.0.attn_q.weight"]

    # output decodes to finite weights of the right shape
    for name, tt, ne0, nrow in SPEC:
        t = [t for t in GGUFReader(out).tensors if t.name == name][0]
        dec = kq.DEQUANTIZE[tt](np.ascontiguousarray(np.asarray(t.data)).reshape(nrow, -1), ne0)
        assert dec.shape == (nrow, ne0) and np.isfinite(dec).all()
