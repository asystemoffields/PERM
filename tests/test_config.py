"""Manifest resumability + onboard divisibility, both without any model download."""
import os

from coldpress.config import Workdir, Manifest
from coldpress.cli import onboard_report


def test_manifest_resumability(tmp_path):
    wd = Workdir(str(tmp_path))
    mf = Manifest(wd)
    out = os.path.join(wd.dir("f16"), "model.gguf")
    open(out, "w").write("x")
    inp = os.path.join(wd.dir("model"), "in.txt")
    open(inp, "w").write("v1")
    ins = {"src": inp, "preset": "Q2_K"}

    assert not mf.is_current("stage", ins, [out])   # never recorded
    mf.record("stage", ins, [out])
    assert mf.is_current("stage", ins, [out])       # recorded, inputs match, output exists

    # a fresh Manifest reads the persisted file
    assert Manifest(wd).is_current("stage", ins, [out])

    # changing a param invalidates
    assert not mf.is_current("stage", {"src": inp, "preset": "Q3_K"}, [out])

    # changing the input file content (fingerprint) invalidates
    import time
    time.sleep(0.01)
    open(inp, "w").write("v2 changed and longer")
    assert not mf.is_current("stage", ins, [out])

    # a missing output invalidates even if inputs match
    mf.record("stage", ins, [out])
    os.remove(out)
    assert not mf.is_current("stage", ins, [out])


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_onboard_divisibility_pass():
    cfg = _Cfg(hidden_size=1024, intermediate_size=3072, num_attention_heads=16,
               num_key_value_heads=8, head_dim=128, model_type="qwen3",
               tie_word_embeddings=True)
    rep = onboard_report(cfg)
    assert rep["divisible_by_256"] is True
    assert rep["tier"] == 2 and rep["spacemap"] == "qwen3"


def test_onboard_divisibility_fail():
    cfg = _Cfg(hidden_size=1000, intermediate_size=3072, num_attention_heads=8,
               head_dim=100, model_type="somearch")
    rep = onboard_report(cfg)
    assert rep["divisible_by_256"] is False
    assert "d_model" in rep["bad_dims"]
    assert rep["tier"] == 1 and rep["spacemap"] is None
