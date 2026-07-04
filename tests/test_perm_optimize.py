"""Tier-2 numeric path: qwen3.optimize produces valid permutations that do not worsen the
container objective, and permute_imatrix moves every .in_sum2 along the right ne[0] axis
(including the composed attn_output index). No llama.cpp, no downloads."""
import numpy as np
import gguf
from gguf import GGUFWriter, GGUFReader

from coldpress.perm.spacemaps import qwen3
from coldpress.perm import optimize as popt, imatrix as pim


class _Cfg:
    hidden_size = 256
    intermediate_size = 512
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 64


D, FFN, V, HD, NKV, NH = 256, 512, 512, 64, 2, 4


def _tensors():
    t = {"token_embd.weight": (V, D), "output.weight": (V, D)}
    for l in range(2):
        t[f"blk.{l}.attn_q.weight"] = (NH * HD, D)
        t[f"blk.{l}.attn_k.weight"] = (NKV * HD, D)
        t[f"blk.{l}.attn_v.weight"] = (NKV * HD, D)
        t[f"blk.{l}.attn_output.weight"] = (D, NH * HD)
        t[f"blk.{l}.ffn_gate.weight"] = (FFN, D)
        t[f"blk.{l}.ffn_up.weight"] = (FFN, D)
        t[f"blk.{l}.ffn_down.weight"] = (D, FFN)
    return t


def _write_f16(path):
    rng = np.random.default_rng(0)
    w = GGUFWriter(path, "qwen3")
    w.add_uint32("qwen3.block_count", 2)
    for n, s in _tensors().items():
        w.add_tensor(n, (rng.standard_normal(s) ** 3 * 0.1).astype(np.float16))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_optimize_valid_and_nonworsening(tmp_path):
    dims = qwen3.dims_from_config(_Cfg())
    f16 = str(tmp_path / "f16.gguf")
    _write_f16(f16)
    typemap = {n: {"type": "Q2_K"} for n in _tensors()}
    perms, report = popt.run_perm(qwen3, f16, typemap, dims, imatrix_path=None,
                                  rows_sample=4096, log=lambda *_: None)
    assert np.array_equal(np.sort(perms["res"]), np.arange(D))
    for l in range(2):
        assert np.array_equal(np.sort(perms["ffn"][l]), np.arange(FFN))
        for h in range(NKV):
            assert np.array_equal(np.sort(perms["vo"][l][h]), np.arange(HD))
    # the chosen permutation never worsens the objective (identity is always in the running)
    for space in ("res", "ffn", "vo"):
        assert report[space]["rel"] <= 1e-9, f"{space} objective worsened: {report[space]}"


def test_permute_imatrix_moves_in_sum2(tmp_path):
    dims = qwen3.dims_from_config(_Cfg())
    perms = {
        "res": np.random.default_rng(1).permutation(D),
        "ffn": [np.random.default_rng(l + 2).permutation(FFN) for l in range(2)],
        "vo": [[np.random.default_rng(10 * l + h).permutation(HD) for h in range(NKV)]
               for l in range(2)],
    }
    rng = np.random.default_rng(3)
    src = str(tmp_path / "imat.gguf")
    w = GGUFWriter(src, "qwen3")
    w.add_uint32("qwen3.block_count", 2)
    entries = 0
    for l in range(2):
        for kind, ne0 in [("attn_q", D), ("attn_k", D), ("attn_v", D),
                          ("attn_output", NH * HD), ("ffn_gate", D), ("ffn_up", D),
                          ("ffn_down", FFN)]:
            base = f"blk.{l}.{kind}.weight"
            w.add_tensor(base + ".in_sum2", (rng.random(ne0) + 0.1).astype(np.float32))
            w.add_tensor(base + ".counts", np.array([100.0], np.float32))
            entries += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    dst = str(tmp_path / "imat-perm.gguf")
    n = pim.permute_imatrix(qwen3, src, dst, perms, dims, log=lambda *_: None)
    assert n == entries

    sv = {t.name: np.array(t.data) for t in GGUFReader(src).tensors if t.name.endswith(".in_sum2")}
    dv = {t.name: np.array(t.data) for t in GGUFReader(dst).tensors if t.name.endswith(".in_sum2")}
    # res-space tensor moves by perms["res"]
    b = "blk.0.attn_q.weight.in_sum2"
    assert np.array_equal(dv[b], sv[b][perms["res"]])
    # ffn_down moves by perms["ffn"][l]
    b = "blk.1.ffn_down.weight.in_sum2"
    assert np.array_equal(dv[b], sv[b][perms["ffn"][1]])
    # attn_output moves by the composed o-input index
    b = "blk.0.attn_output.weight.in_sum2"
    assert np.array_equal(dv[b], sv[b][qwen3.o_input_index(perms, 0, dims)])
