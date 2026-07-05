"""CLI flag wiring for the split scale chain: --perms-only parses on the perm subcommand
(and ONLY defaults False elsewhere), and the scale flags land in RunConfig."""
from coldpress import cli


def test_perms_only_parses_on_perm():
    ap = cli.build_parser()
    a = ap.parse_args(["perm", "X", "--workdir", "w", "--perms-only"])
    assert a.perms_only is True
    a = ap.parse_args(["perm", "X", "--workdir", "w"])
    assert a.perms_only is False
    # other subcommands don't define it; cmd_perm reads it via getattr(default False)
    q = ap.parse_args(["quantize", "X", "--workdir", "w"])
    assert getattr(q, "perms_only", False) is False


def test_scale_flags_reach_runconfig():
    ap = cli.build_parser()
    a = ap.parse_args(["calibrate", "X", "--workdir", "w", "--teacher-dtype", "bfloat16",
                       "--teacher-device", "auto", "--hessian-layer-range", "24:48"])
    cfg = cli._cfg_from_args(a)
    assert cfg.teacher_dtype == "bfloat16"
    assert cfg.teacher_device == "auto"
    assert cfg.hessian_layer_range == (24, 48)


# ---------------------------------------------------------------- multimodal config seam

class _Txt:
    """A qwen3.5-9B-shaped text config."""
    hidden_size = 4096
    intermediate_size = 12288
    num_hidden_layers = 32
    vocab_size = 248320
    num_attention_heads = 16
    num_key_value_heads = 4
    head_dim = 256
    tie_word_embeddings = False
    model_type = "qwen3_5_text"


class _Wrapper:
    """A multimodal wrapper config: dims nest under text_config, model_type stays top-level."""
    text_config = _Txt()
    model_type = "qwen3_5"


def test_text_config_helper_unwraps():
    from coldpress.config import text_config
    w = _Wrapper()
    assert text_config(w) is _Wrapper.text_config
    t = _Txt()
    assert text_config(t) is t                      # plain text config passes through

    class _NoneWrap:
        text_config = None
    nw = _NoneWrap()
    assert text_config(nw) is nw                    # text_config=None edge -> self


def test_param_estimate_unwraps_wrapper():
    est_wrapped = cli._param_count_estimate(_Wrapper())
    est_plain = cli._param_count_estimate(_Txt())
    assert est_wrapped == est_plain > 3_000_000_000  # 9B-class -> bf16 default


def test_onboard_report_wrapper_dims_and_toplevel_model_type():
    rep = cli.onboard_report(_Wrapper())
    assert rep["model_type"] == "qwen3_5"           # TOP-LEVEL keying preserved (registry)
    assert rep["tier"] == 2                          # qwen3_5 -> qwen35 spacemap
    assert rep["dims"]["d_model"] == 4096            # dims from the NESTED text stack
    assert rep["dims"]["d_ffn"] == 12288
    assert rep["divisible_by_256"]
    assert rep["tie_word_embeddings"] is False
