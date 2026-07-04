"""qwen35 + gemma4 spacemaps (SCALE blocker #3). The REAL verification: random perms pushed
through apply_perms (via the g3_check reload path) preserve logits on a tiny random model of
each arch -- the space maps are function-preserving. Plus save/load perms round-trip (vo is a
DICT keyed by layer), optimize produces valid non-worsening perms, and the review guard bites.

Tiny configs (seconds, no downloads):
  qwen35: 2 layers, layer 0 = linear_attention (GatedDeltaNet, P_lav frozen identity),
          layer 1 = full_attention (doubled q_proj query+gate -- the GATE TRAP). head_dim is
          the map's hardcoded _HD=256; the linear-attn path runs on CPU via the torch fallback.
  gemma4: 2 layers, layer 0 = sliding_attention (P_vo), layer 1 = full_attention (K=V shared,
          no v_proj -> global P_vo frozen identity). TIED embeddings. The config forces the
          LAST layer to full_attention, so a global layer is always exercised.
"""
import copy

import numpy as np
import pytest

from coldpress.perm.spacemaps import qwen35, gemma4

ACK = dict(acknowledge_unreviewed=True)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def tiny_qwen35():
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=256,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        max_position_embeddings=128, tie_word_embeddings=False, pad_token_id=0,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).eval(), cfg


@pytest.fixture(scope="session")
def tiny_gemma4():
    import torch
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    cfg = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=256, global_head_dim=256,
        num_global_key_value_heads=1, attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0, enable_moe_block=False, num_kv_shared_layers=0,
        max_position_embeddings=128, tie_word_embeddings=True, sliding_window=64,
    )
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg).eval(), cfg


@pytest.fixture(scope="session")
def ids16():
    import torch
    torch.manual_seed(1)
    return torch.randint(0, 60, (1, 16))


def _random_perms(sm, dims, seed):
    rng = np.random.default_rng(seed)
    vo_layers = dims.get("full_attn_layers") if sm is qwen35 else dims.get("sliding_layers")
    return {
        "res": rng.permutation(dims["d_model"]),
        "ffn": [rng.permutation(dims["d_ffn"]) for _ in range(dims["n_layers"])],
        "vo": {l: [rng.permutation(dims["head_dim"]) for _ in range(dims["n_kv"])]
               for l in vo_layers},
    }


# ---------------------------------------------------------------- G3 (the headline)

def test_qwen35_g3_identity_and_random(tiny_qwen35, ids16):
    model, cfg = tiny_qwen35
    dims = qwen35.dims_from_config(cfg, **ACK)
    d0, _ = qwen35.g3_check(copy.deepcopy(model), qwen35.identity_perms(dims, **ACK), dims,
                            ids16, **ACK)
    assert d0 < 1e-4, f"identity perms drifted logits by {d0}"
    d, rel = qwen35.g3_check(copy.deepcopy(model), _random_perms(qwen35, dims, 3), dims,
                             ids16, **ACK)
    assert rel < 1e-4, f"qwen35 G3 FAIL: rel={rel:.3e} max|dlogit|={d:.3e}"


def test_gemma4_g3_identity_and_random(tiny_gemma4, ids16):
    model, cfg = tiny_gemma4
    dims = gemma4.dims_from_config(cfg, **ACK)
    d0, _ = gemma4.g3_check(copy.deepcopy(model), gemma4.identity_perms(dims, **ACK), dims,
                            ids16, **ACK)
    assert d0 < 1e-4, f"identity perms drifted logits by {d0}"
    d, rel = gemma4.g3_check(copy.deepcopy(model), _random_perms(gemma4, dims, 5), dims,
                             ids16, **ACK)
    assert rel < 1e-4, f"gemma4 G3 FAIL: rel={rel:.3e} max|dlogit|={d:.3e}"


# ---------------------------------------------------------------- save/load (vo dict)

@pytest.mark.parametrize("sm", [qwen35, gemma4])
def test_save_load_perms_roundtrip(sm, tmp_path):
    dims = {"d_model": 256, "d_ffn": 512, "n_layers": 3, "n_kv": 2, "head_dim": 256}
    vo_layers = [1]   # only some layers carry P_vo (full/sliding)
    perms = {
        "res": np.random.default_rng(0).permutation(256),
        "ffn": [np.random.default_rng(l + 1).permutation(512) for l in range(3)],
        "vo": {l: [np.random.default_rng(10 * l + h).permutation(256) for h in range(2)]
               for l in vo_layers},
    }
    p = str(tmp_path / "perms.npz")
    sm.save_perms(perms, p, dims, **ACK)
    out = sm.load_perms(p, **ACK)
    assert np.array_equal(out["res"], perms["res"])
    for l in range(3):
        assert np.array_equal(out["ffn"][l], perms["ffn"][l])
    assert sorted(out["vo"]) == vo_layers
    for l in vo_layers:
        for h in range(2):
            assert np.array_equal(out["vo"][l][h], perms["vo"][l][h])


# ---------------------------------------------------------------- optimize validity

def _synth_weights(names, ne0_of, seed=0):
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal((max(ne0_of[n], 256), ne0_of[n])) ** 3 * 0.1).astype(np.float16)
            for n in names}


def test_qwen35_optimize_valid_and_nonworsening():
    dims = {"d_model": 256, "d_ffn": 512, "n_layers": 2, "n_kv": 2, "n_heads": 4,
            "head_dim": 256, "full_attn_layers": [1]}
    D, FFN, AO = 256, 512, 4 * 256
    names = {"token_embd.weight": D, "output.weight": D}
    for l in range(2):
        names[f"blk.{l}.ffn_gate.weight"] = D
        names[f"blk.{l}.ffn_up.weight"] = D
        names[f"blk.{l}.ffn_down.weight"] = FFN
    names["blk.1.attn_q.weight"] = D
    names["blk.1.attn_k.weight"] = D
    names["blk.1.attn_v.weight"] = D
    names["blk.1.attn_output.weight"] = AO
    weights = _synth_weights(list(names), names)
    ttypes = {n: "Q2_K" for n in names}
    perms, report = qwen35.optimize(weights, ttypes, {}, dims, rows_sample=4096,
                                    log=lambda *_: None, **ACK)
    assert np.array_equal(np.sort(perms["res"]), np.arange(D))
    for l in range(2):
        assert np.array_equal(np.sort(perms["ffn"][l]), np.arange(FFN))
    assert list(perms["vo"]) == [1]
    for h in range(2):
        assert np.array_equal(np.sort(perms["vo"][1][h]), np.arange(256))
    for space in ("res", "ffn", "vo"):
        assert report[space]["rel"] <= 1e-9, f"{space} worsened: {report[space]}"


def test_gemma4_optimize_valid_and_nonworsening():
    dims = {"d_model": 256, "d_ffn": 512, "n_layers": 2, "n_kv": 2, "n_heads": 4,
            "head_dim": 256, "sliding_layers": [0], "global_layers": [1]}
    D, FFN, AO = 256, 512, 4 * 256
    names = {"token_embd.weight": D}
    for l in range(2):
        names[f"blk.{l}.attn_q.weight"] = D
        names[f"blk.{l}.attn_k.weight"] = D
        names[f"blk.{l}.ffn_gate.weight"] = D
        names[f"blk.{l}.ffn_up.weight"] = D
        names[f"blk.{l}.ffn_down.weight"] = FFN
    names["blk.0.attn_v.weight"] = D           # sliding layer has v_proj
    names["blk.0.attn_output.weight"] = AO
    names["blk.1.attn_output.weight"] = AO
    weights = _synth_weights(list(names), names)
    ttypes = {n: "Q2_K" for n in names}
    perms, report = gemma4.optimize(weights, ttypes, {}, dims, rows_sample=4096,
                                    log=lambda *_: None, **ACK)
    assert np.array_equal(np.sort(perms["res"]), np.arange(D))
    assert list(perms["vo"]) == [0]            # sliding only; global frozen identity
    for h in range(2):
        assert np.array_equal(np.sort(perms["vo"][0][h]), np.arange(256))
    for space in ("res", "ffn", "vo"):
        assert report[space]["rel"] <= 1e-9, f"{space} worsened: {report[space]}"


# ---------------------------------------------------------------- permute_imatrix

@pytest.mark.parametrize("sm,arch,vo_key", [(qwen35, "qwen35", "full_attn_layers"),
                                            (gemma4, "gemma4", "sliding_layers")])
def test_permute_imatrix_moves_in_sum2(sm, arch, vo_key, tmp_path):
    """A recognized res-space in_sum2 moves by P_res; ffn_down by P_ffn; the vo-carrying
    attn_output by the composed o-input index."""
    from gguf import GGUFWriter, GGUFReader
    D, FFN, HD, NKV, NH = 256, 512, 256, 2, 4
    dims = {"d_model": D, "d_ffn": FFN, "n_layers": 2, "n_kv": NKV, "n_heads": NH,
            "head_dim": HD, "full_attn_layers": [1], "sliding_layers": [0], "global_layers": [1]}
    rng = np.random.default_rng(1)
    vo_layer = dims[vo_key][0]
    perms = {"res": rng.permutation(D),
             "ffn": [rng.permutation(FFN) for _ in range(2)],
             "vo": {vo_layer: [rng.permutation(HD) for _ in range(NKV)]}}
    src = str(tmp_path / "imat.gguf")
    w = GGUFWriter(src, arch)
    w.add_uint32(f"{arch}.block_count", 2)
    tens = {"blk.0.attn_q.weight": D, "blk.1.ffn_down.weight": FFN,
            f"blk.{vo_layer}.attn_output.weight": NH * HD}
    rr = np.random.default_rng(3)
    for base, ne0 in tens.items():
        w.add_tensor(base + ".in_sum2", (rr.random(ne0) + 0.1).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    dst = str(tmp_path / "imat-perm.gguf")
    n = sm.permute_imatrix(src, dst, perms, dims, **ACK)
    assert n == 3
    sv = {t.name: np.array(t.data) for t in GGUFReader(src).tensors}
    dv = {t.name: np.array(t.data) for t in GGUFReader(dst).tensors}
    assert np.array_equal(dv["blk.0.attn_q.weight.in_sum2"],
                          sv["blk.0.attn_q.weight.in_sum2"][perms["res"]])
    assert np.array_equal(dv["blk.1.ffn_down.weight.in_sum2"],
                          sv["blk.1.ffn_down.weight.in_sum2"][perms["ffn"][1]])
    ao = f"blk.{vo_layer}.attn_output.weight"
    idx = sm.input_perm(ao, perms, dims, **ACK)
    assert np.array_equal(dv[ao + ".in_sum2"], sv[ao + ".in_sum2"][np.asarray(idx)])


# ---------------------------------------------------------------- review guard

@pytest.mark.parametrize("sm", [qwen35, gemma4])
def test_guard_bites_without_ack(sm):
    with pytest.raises(NotImplementedError, match="PENDING FABLE REVIEW"):
        sm.identity_perms({"d_model": 8, "d_ffn": 16, "n_layers": 1, "n_kv": 1,
                           "head_dim": 256, "full_attn_layers": [0], "sliding_layers": [0]})
    with pytest.raises(NotImplementedError, match="PENDING FABLE REVIEW"):
        sm.optimize({}, {}, {}, {"d_model": 8, "d_ffn": 16, "n_layers": 0, "n_kv": 1,
                                 "n_heads": 1, "head_dim": 256, "full_attn_layers": [],
                                 "sliding_layers": []})
