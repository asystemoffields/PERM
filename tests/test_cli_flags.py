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
