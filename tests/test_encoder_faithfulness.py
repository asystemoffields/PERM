"""Encoder faithfulness vs REAL llama-quantize (skipif no llama.cpp).

Builds a minimal f16 GGUF, quantizes it with the actual llama-quantize binary (ref path,
no imatrix), and checks our kquant encoder is faithful. Two notions of faithful:
  * reconstruction quality (max_err_delta): our imatrix-weighted quant error must match
    llama.cpp's to <1% -- the invariant that MATTERS, asserted here on every preset.
  * raw byte-diff: on REAL model weights this is <0.1% median (METHOD.md: 0.028% across 198
    tensors). On SYNTHETIC gaussian^3 weights, near-tie scale decisions flip far more often
    between numpy's pairwise sum and C's sequential fp32 loop, so the raw byte-diff is only
    meaningful on real weights -- we do NOT gate on it here.
Self-contained: no model download."""
import numpy as np
import pytest
import gguf
from gguf import GGUFWriter

from coldpress import llamacpp, gates
from conftest import heavy_tailed

LLAMA = llamacpp.locate()
pytestmark = pytest.mark.skipif(LLAMA is None,
                                reason="llama.cpp not found (set COLDPRESS_LLAMACPP or PATH)")


def _mini_f16(path, seed=0):
    d, ffn, V, hd, nkv, nh = 256, 512, 512, 64, 2, 4
    rng = np.random.default_rng(seed)

    def wf(shape):
        # heavy-tailed like real weights (fewer near-tie scale flips)
        return (rng.standard_normal(shape) ** 3 * 0.05).astype(np.float16)

    w = GGUFWriter(path, "llama")
    w.add_uint32("llama.block_count", 1)
    w.add_uint32("llama.context_length", 128)
    w.add_uint32("llama.embedding_length", d)
    w.add_uint32("llama.feed_forward_length", ffn)
    w.add_uint32("llama.attention.head_count", nh)
    w.add_uint32("llama.attention.head_count_kv", nkv)
    w.add_uint32("llama.vocab_size", V)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_tensor("token_embd.weight", wf((V, d)))
    w.add_tensor("output.weight", wf((V, d)))
    w.add_tensor("output_norm.weight", np.ones(d, np.float32))
    w.add_tensor("blk.0.attn_q.weight", wf((d, d)))
    w.add_tensor("blk.0.attn_k.weight", wf((nkv * hd, d)))
    w.add_tensor("blk.0.attn_v.weight", wf((nkv * hd, d)))
    w.add_tensor("blk.0.attn_output.weight", wf((d, nh * hd)))
    w.add_tensor("blk.0.attn_norm.weight", np.ones(d, np.float32))
    w.add_tensor("blk.0.ffn_gate.weight", wf((ffn, d)))
    w.add_tensor("blk.0.ffn_up.weight", wf((ffn, d)))
    w.add_tensor("blk.0.ffn_down.weight", wf((d, ffn)))
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(d, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


@pytest.mark.parametrize("preset", ["Q2_K", "Q3_K_S", "Q4_K", "Q6_K"])
def test_encoder_faithful_to_llama_quantize(preset, tmp_path):
    f16 = str(tmp_path / "mini-f16.gguf")
    stock = str(tmp_path / f"mini-{preset}.gguf")
    _mini_f16(f16)
    LLAMA.quantize(f16, stock, preset, logfile=str(tmp_path / "q.log"))
    # gate on reconstruction quality; relax raw byte-diff (synthetic near-tie flips).
    passed, rep = gates.encoder_faithfulness(f16, stock, imatrix_path=None,
                                             max_byte_diff=1.0, max_err_delta=0.01,
                                             log=lambda *_: None)
    assert rep["max_err_delta"] < 0.01, \
        f"{preset}: our quant error deviates {rep['max_err_delta']*100:.3f}% from llama-quantize"
    assert rep["tested"] > 0
