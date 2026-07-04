#!/usr/bin/env python3
"""coldpress CLI. Subcommands (see the contract):

  onboard <hf_id>     config fetch, divisibility check, arch/tier report, plan printout
  calibrate           download model rev; build/locate llama.cpp; f16 convert; imatrix;
                      Hessians (generic collector, --hessian-shards N); teacher top-K cache
  perm                (tier 2) optimize perms; apply to a copy; G3 gate; permute imatrix;
                      convert permuted -> f16
  encode              stock-quantize at preset; EF (act_order always on); byte-parity gate
  distill             --norm (fast) and/or --scales (full E3B); untied student
  verify              re-read final GGUF; typemap gate; smoke gen; optional --ppl-check
  quantize            onboard -> calibrate -> [perm] -> encode -> [distill] -> verify
  build-llamacpp      clone the pinned commit and build the required binaries

Every stage writes a manifest entry (input fingerprints, versions, timestamps) into
--workdir and is skipped when its inputs still match and its outputs exist.
"""
import argparse
import os
import sys

from . import __version__, llamacpp
from .config import RunConfig, Workdir, Manifest
from .perm import registry


def _log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- onboard

# candidate matmul ne[0] values that MUST be divisible by 256 for k-quants
def onboard_report(config):
    def g(*names, default=None):
        for n in names:
            v = getattr(config, n, None)
            if v is not None:
                return v
        return default
    d_model = g("hidden_size")
    n_heads = g("num_attention_heads")
    head_dim = g("head_dim", default=(d_model // n_heads if d_model and n_heads else None))
    d_ffn = g("intermediate_size")
    moe_ffn = g("moe_intermediate_size", "expert_intermediate_size")
    dims = {
        "d_model": d_model, "d_ffn": d_ffn, "n_heads": n_heads,
        "head_dim": head_dim, "attn_out": (n_heads * head_dim) if (n_heads and head_dim) else None,
        "moe_ffn": moe_ffn,
    }
    matmul_ne0 = {k: v for k, v in
                  {"d_model": d_model, "d_ffn": d_ffn, "attn_out": dims["attn_out"],
                   "moe_ffn": moe_ffn}.items() if v is not None}
    bad = {k: v for k, v in matmul_ne0.items() if v % 256 != 0}
    model_type = getattr(config, "model_type", None)
    spacemap = registry.get_spacemap(model_type)
    return {
        "model_type": model_type,
        "dims": dims,
        "matmul_ne0": matmul_ne0,
        "divisible_by_256": not bad,
        "bad_dims": bad,
        "tier": 2 if spacemap is not None else 1,
        "spacemap": model_type if spacemap is not None else None,
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
    }


def cmd_onboard(cfg, args):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(cfg.hf_id, revision=cfg.revision,
                                        trust_remote_code=args.trust_remote_code)
    if hasattr(config, "text_config") and getattr(config, "model_type", "") in (
            "gemma4", "gemma3", "qwen2_vl", "qwen3_vl"):
        pass  # keep the top-level model_type for spacemap keying
    rep = onboard_report(config)
    _log(f"== onboard {cfg.hf_id} (rev {cfg.revision or 'default'}) ==")
    _log(f"  model_type : {rep['model_type']}")
    _log(f"  dims       : {rep['dims']}")
    _log(f"  matmul ne0 : {rep['matmul_ne0']}")
    if rep["divisible_by_256"]:
        _log("  [OK] all matmul ne[0] divisible by 256")
    else:
        _log(f"  [STOP] not divisible by 256: {rep['bad_dims']}  -- k-quant containers need "
             f"256-multiples; this arch/size is unsupported")
    if rep["tier"] == 2:
        _log(f"  tier 2: spacemap '{rep['spacemap']}' -> PERM + EF + NORM/E3B available")
    else:
        _log(f"  tier 1: no spacemap for '{rep['model_type']}' -> EF + NORM/E3B only "
             f"(PERM skipped). Available spacemaps: {registry.available()}. "
             f"To add one, see docs/spacemaps.md (RECIPE.md §10).")
    return rep


# ---------------------------------------------------------------- llama.cpp

def _get_llama(cfg, require=True):
    lc = llamacpp.locate(cfg.llama_cpp)
    if lc is None and require:
        raise SystemExit(
            "llama.cpp not found. Pass --llama-cpp DIR, set COLDPRESS_LLAMACPP, put "
            "llama-quantize on PATH, or run `coldpress build-llamacpp --dest DIR`"
            + (" --cuda" if cfg.cuda else "") + ".")
    return lc


def cmd_build_llamacpp(cfg, args):
    lc = llamacpp.build(args.dest, cuda=args.cuda or cfg.cuda, cuda_arch=cfg.cuda_arch,
                        log=_log)
    _log(f"built llama.cpp @ {llamacpp.PINNED_COMMIT} -> {lc.bin_dir}")
    _log(f"export COLDPRESS_LLAMACPP={os.path.dirname(lc.bin_dir) if lc.repo_dir is None else lc.repo_dir}")
    return lc


# ---------------------------------------------------------------- calibrate

def cmd_calibrate(cfg, args, wd=None, mf=None, llama=None):
    import numpy as np
    wd = wd or Workdir(cfg.workdir)
    mf = mf or Manifest(wd)
    llama = llama or _get_llama(cfg)
    calib = cfg.resolved_calib()

    # 1. model snapshot
    model_dir = wd.dir("model")
    ins = {"hf_id": cfg.hf_id, "revision": cfg.revision}
    marker = os.path.join(model_dir, ".downloaded")
    if not mf.is_current("model", ins, [marker]):
        from huggingface_hub import snapshot_download
        snapshot_download(cfg.hf_id, revision=cfg.revision, local_dir=model_dir)
        open(marker, "w").write(cfg.revision or "default")
        mf.record("model", ins, [marker])
    _log(f"model: {model_dir}")

    # 2. f16 convert
    f16 = os.path.join(wd.dir("f16"), "model-f16.gguf")
    ins = {"model": marker}
    if not mf.is_current("f16", ins, [f16]):
        llama.convert_hf_to_gguf(model_dir, f16, outtype="f16",
                                 logfile=wd.log("convert.log"))
        mf.record("f16", ins, [f16], llama_commit=llamacpp.PINNED_COMMIT)
    _log(f"f16: {f16}")

    # 3. imatrix
    imatrix = os.path.join(wd.dir("imatrix"), "imatrix.gguf")
    ins = {"f16": f16, "calib": calib}
    if not mf.is_current("imatrix", ins, [imatrix]):
        llama.imatrix(f16, calib, imatrix, ctx=512, ngl=99 if cfg.cuda else 0,
                      logfile=wd.log("imatrix.log"))
        mf.record("imatrix", ins, [imatrix], llama_commit=llamacpp.PINNED_COMMIT)
    _log(f"imatrix: {imatrix}")

    # 4. Hessians (generic collector, sharded)
    from . import hessians as hess
    from .ggufio import read_typemap
    hdir = wd.dir("hessians")
    ins = {"f16": f16, "calib": calib, "shards": cfg.hessian_shards,
           "chunks": cfg.n_calib_chunks}
    marker = os.path.join(hdir, ".done")
    if not mf.is_current("hessians", ins, [marker]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        tm = read_typemap(f16)
        gguf_ne0 = {n: v["shape"][0] for n, v in tm.items()}
        arch = _arch_of(f16)
        tok = AutoTokenizer.from_pretrained(model_dir)
        ids = tok(open(calib, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
        ctx = 512
        n = min(cfg.n_calib_chunks, len(ids) // ctx)
        batches = [ids[c * ctx:(c + 1) * ctx] for c in range(n)]
        n_layers = AutoConfig.from_pretrained(model_dir).num_hidden_layers
        shards = _shard_ranges(n_layers, cfg.hessian_shards)
        for lo, hi in shards:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
            model.eval()
            hs, unmapped = hess.collect_hessians(model, batches, gguf_ne0, arch, n_layers,
                                                 layer_lo=lo, layer_hi=hi, log=_log)
            hess.save_hessians(hs, hdir)
            del model
        open(marker, "w").write("ok")
        mf.record("hessians", ins, [marker])
    _log(f"hessians: {hdir}")

    # 5. teacher top-K cache
    from . import teacher as tea
    tdir = wd.dir("teacher")
    ins = {"model": os.path.join(model_dir, ".downloaded"), "calib": calib,
           "chunks": cfg.n_calib_chunks}
    marker = os.path.join(tdir, ".done")
    if not mf.is_current("teacher", ins, [marker]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        tea.cache_teacher(model, tok, tdir, calib, n_chunks=cfg.n_calib_chunks, log=_log)
        del model
        open(marker, "w").write("ok")
        mf.record("teacher", ins, [marker])
    _log(f"teacher: {tdir}")
    return {"model_dir": model_dir, "f16": f16, "imatrix": imatrix,
            "hessians": hdir, "teacher": tdir}


def _shard_ranges(n_layers, shards):
    shards = max(1, shards)
    step = (n_layers + shards - 1) // shards
    return [(lo, min(lo + step, n_layers)) for lo in range(0, n_layers, step)]


def _arch_of(gguf_path):
    from gguf import GGUFReader
    from .ggufio import read_arch
    return read_arch(GGUFReader(gguf_path))


# ---------------------------------------------------------------- perm

def cmd_perm(cfg, args, wd=None, mf=None, llama=None, cal=None):
    wd = wd or Workdir(cfg.workdir)
    mf = mf or Manifest(wd)
    llama = llama or _get_llama(cfg)
    cal = cal or {"model_dir": wd.dir("model"), "f16": os.path.join(wd.dir("f16"), "model-f16.gguf"),
                  "imatrix": os.path.join(wd.dir("imatrix"), "imatrix.gguf")}
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    from .perm import optimize as perm_opt, imatrix as perm_im
    from . import gates
    from .ggufio import read_typemap

    config = AutoConfig.from_pretrained(cal["model_dir"])
    model_type = getattr(config, "model_type", None)
    sm = registry.get_spacemap(model_type)
    if sm is None:
        _log(f"[perm] no spacemap for '{model_type}' -> skipping PERM (tier 1). "
             f"Available: {registry.available()}")
        return None
    ack = {}
    if getattr(sm, "MODEL_TYPE", None) in ("qwen35", "gemma4"):
        if not cfg.acknowledge_unreviewed:
            _log(f"[perm] spacemap '{sm.MODEL_TYPE}' is PENDING FABLE REVIEW; pass "
                 f"--acknowledge-unreviewed to run it anyway. Skipping PERM.")
            return None
        ack = {"acknowledge_unreviewed": True}
    dims = sm.dims_from_config(config, **ack)

    # target typemap = stock quant at preset
    stock = os.path.join(wd.dir("encode"), f"stock-{cfg.preset}.gguf")
    ins = {"f16": cal["f16"], "imatrix": cal["imatrix"], "preset": cfg.preset}
    if not mf.is_current("stock_quant", ins, [stock]):
        llama.quantize(cal["f16"], stock, cfg.preset, imatrix=cal["imatrix"],
                       logfile=wd.log("stock-quant.log"))
        mf.record("stock_quant", ins, [stock], llama_commit=llamacpp.PINNED_COMMIT)
    typemap = read_typemap(stock)

    # optimize perms
    perms_npz = os.path.join(wd.dir("perm"), "perms.npz")
    ins = {"f16": cal["f16"], "typemap_of": stock, "imatrix": cal["imatrix"],
           "rows_sample": cfg.rows_sample}
    if not mf.is_current("perm_optimize", ins, [perms_npz]):
        perms, report = perm_opt.run_perm(sm, cal["f16"], typemap, dims,
                                          imatrix_path=cal["imatrix"],
                                          rows_sample=cfg.rows_sample, log=_log)
        sm.save_perms(perms, perms_npz, dims)
        mf.record("perm_optimize", ins, [perms_npz], extra=report)
    perms = sm.load_perms(perms_npz)

    # G3 gate on the real model
    tok_ids = _calib_ids(cal["model_dir"], cfg.resolved_calib(), 256)
    model = AutoModelForCausalLM.from_pretrained(cal["model_dir"], dtype=torch.float32)
    d, rel = gates.g3(sm, model, perms, dims, tok_ids, **ack)
    _log(f"[perm] G3 PASS: rel logits drift {rel:.3e} (max|dlogit|={d:.3e})")
    del model

    # apply perms to a copy of the checkpoint, convert -> f16
    pdir = wd.dir("perm")
    permuted_model = os.path.join(pdir, "model-permuted")
    _write_permuted_checkpoint(sm, cal["model_dir"], permuted_model, perms, dims, ack)
    perm_f16 = os.path.join(pdir, "model-permuted-f16.gguf")
    ins = {"permuted": permuted_model}
    if not mf.is_current("perm_f16", ins, [perm_f16]):
        llama.convert_hf_to_gguf(permuted_model, perm_f16, outtype="f16",
                                 logfile=wd.log("convert-perm.log"))
        mf.record("perm_f16", ins, [perm_f16], llama_commit=llamacpp.PINNED_COMMIT)

    # permute imatrix
    perm_imat = os.path.join(pdir, "imatrix-perm.gguf")
    ins = {"imatrix": cal["imatrix"], "perms": perms_npz}
    if not mf.is_current("perm_imatrix", ins, [perm_imat]):
        perm_im.permute_imatrix(sm, cal["imatrix"], perm_imat, perms, dims, log=_log)
        mf.record("perm_imatrix", ins, [perm_imat])
    return {"spacemap": sm, "perms": perms, "perms_npz": perms_npz, "dims": dims,
            "f16": perm_f16, "imatrix": perm_imat, "model_dir": permuted_model}


def _calib_ids(model_dir, calib, n_tok):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = open(calib, encoding="utf-8").read()[:20000]
    return tok(text, return_tensors="pt").input_ids[:, :n_tok]


def _write_permuted_checkpoint(sm, src_dir, out_dir, perms, dims, ack):
    import shutil
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    # load state dict (single-file safetensors expected for small models)
    st = os.path.join(src_dir, "model.safetensors")
    sd = load_file(st)
    out = sm.apply_perms(sd, perms, dims, consume=True, **ack)
    save_file(out, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    for f in ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
        s = os.path.join(src_dir, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(out_dir, f))
    return out_dir


# ---------------------------------------------------------------- encode

def cmd_encode(cfg, args, wd=None, mf=None, llama=None, source=None):
    """source: dict from perm (permuted f16/imatrix/spacemap/perms/dims) or None (tier 1,
    stock orientation from calibrate outputs)."""
    wd = wd or Workdir(cfg.workdir)
    mf = mf or Manifest(wd)
    llama = llama or _get_llama(cfg)
    from . import ef, gates

    if source:
        f16, imatrix = source["f16"], source["imatrix"]
        sm, perms, dims = source["spacemap"], source["perms"], source["dims"]
        tag = "permef"
    else:
        f16 = os.path.join(wd.dir("f16"), "model-f16.gguf")
        imatrix = os.path.join(wd.dir("imatrix"), "imatrix.gguf")
        sm = perms = dims = None
        tag = "ef"
    if imatrix and not os.path.exists(imatrix):
        imatrix = None  # calibrate builds it; without it, quantize/EF use the ref path

    # stock quant at preset (the byte-map target)
    stock = os.path.join(wd.dir("encode"), f"stock-{tag}-{cfg.preset}.gguf")
    ins = {"f16": f16, "imatrix": imatrix or "", "preset": cfg.preset}
    if not mf.is_current(f"stock_{tag}", ins, [stock]):
        llama.quantize(f16, stock, cfg.preset, imatrix=imatrix,
                       logfile=wd.log(f"stock-{tag}.log"))
        mf.record(f"stock_{tag}", ins, [stock], llama_commit=llamacpp.PINNED_COMMIT)

    out = os.path.join(wd.dir("encode"), f"coldpress-{tag}-{cfg.preset}.gguf")
    ins = {"f16": f16, "stock": stock, "hessians": os.path.join(wd.dir("hessians"), ".done"),
           "imatrix": imatrix, "unsafe": args.unsafe_storage_order}
    if not mf.is_current(f"encode_{tag}", ins, [out]):
        ef.encode_gguf(f16, stock, wd.dir("hessians"), out, imatrix_path=imatrix,
                       perms=perms, spacemap=sm, unsafe_storage_order=args.unsafe_storage_order,
                       device=cfg.device, log=_log)
        mf.record(f"encode_{tag}", ins, [out])
    # byte-parity gate
    rep = gates.byte_parity(out, stock)
    _log(f"[encode] byte-parity PASS: {rep['tensors']} tensors, "
         f"{rep['total_final']} <= {rep['total_stock']} bytes")
    return {"gguf": out, "stock": stock, "tag": tag}


# ---------------------------------------------------------------- distill

def cmd_distill(cfg, args, wd=None, mf=None, enc=None):
    wd = wd or Workdir(cfg.workdir)
    mf = mf or Manifest(wd)
    from . import distill
    from transformers import AutoConfig
    model_dir = wd.dir("model")
    config = AutoConfig.from_pretrained(model_dir)
    arch = _arch_of(os.path.join(wd.dir("f16"), "model-f16.gguf"))
    n_layers = config.num_hidden_layers
    teacher_dir = wd.dir("teacher")
    cur = enc["gguf"]

    if cfg.distill_norm:
        out = os.path.join(wd.dir("distill"), os.path.basename(cur).replace(".gguf", "-norm.gguf"))
        ins = {"in": cur, "teacher": os.path.join(teacher_dir, ".done"), "steps": args.norm_steps}
        if not mf.is_current("distill_norm", ins, [out]):
            distill.distill_norm(model_dir, arch, n_layers, cur, teacher_dir, out,
                                 steps=args.norm_steps, log=_log)
            mf.record("distill_norm", ins, [out])
        cur = out
    if cfg.distill_scales:
        out = os.path.join(wd.dir("distill"), os.path.basename(cur).replace(".gguf", "-e3b.gguf"))
        ins = {"in": cur, "teacher": os.path.join(teacher_dir, ".done"), "steps": args.scale_steps}
        if not mf.is_current("distill_scales", ins, [out]):
            distill.distill_e3b(model_dir, arch, n_layers, cur, teacher_dir, out,
                                steps=args.scale_steps, device=cfg.device, log=_log)
            mf.record("distill_scales", ins, [out])
        cur = out
    return {"gguf": cur}


# ---------------------------------------------------------------- verify

def cmd_verify(cfg, args, wd=None, mf=None, llama=None, final=None, stock=None):
    wd = wd or Workdir(cfg.workdir)
    from . import gates
    final = final or args.gguf
    stock = stock or args.stock
    _log(f"== verify {final} ==")
    if stock:
        rep = gates.byte_parity(final, stock)
        _log(f"  byte-parity PASS: {rep['tensors']} tensors, "
             f"{rep['total_final']} <= {rep['total_stock']} bytes")
    llama = llama or _get_llama(cfg, require=False)
    if llama is not None:
        out = gates.smoke(llama, final, prompt=args.prompt, ngl=99 if cfg.cuda else 0,
                          logfile=wd.log("smoke.log"))
        _log("  smoke PASS: generated non-empty output")
        if args.ppl_check:
            from . import parse_ppl
            a = wd.log("ppl-stock.log")
            b = wd.log("ppl-final.log")
            if stock:
                llama.perplexity(stock, args.ppl_check, threads=cfg.threads, logfile=a)
            llama.perplexity(final, args.ppl_check, threads=cfg.threads, logfile=b)
            if stock:
                r = parse_ppl.paired_delta(a, b)
                _log(f"  ppl A/B (final vs stock): {r['point_pct']:+.3f}% "
                     f"[95% CI {r['ci95_lo_pct']:+.3f} .. {r['ci95_hi_pct']:+.3f}]")
    else:
        _log("  (llama.cpp not found; skipped smoke + ppl)")
    return {"final": final}


# ---------------------------------------------------------------- quantize (chain)

def cmd_quantize(cfg, args):
    wd = Workdir(cfg.workdir)
    mf = Manifest(wd)
    llama = _get_llama(cfg)
    cmd_onboard(cfg, args)
    cal = cmd_calibrate(cfg, args, wd, mf, llama)
    source = None
    if cfg.perm:
        source = cmd_perm(cfg, args, wd, mf, llama, cal)
    enc = cmd_encode(cfg, args, wd, mf, llama, source=source)
    if cfg.distill_norm or cfg.distill_scales:
        enc = cmd_distill(cfg, args, wd, mf, enc)
    final = enc["gguf"]
    if cfg.out:
        import shutil
        shutil.copy(final, cfg.out)
        final = cfg.out
    cmd_verify(cfg, args, wd, mf, llama, final=final, stock=enc["stock"])
    _log(f"\n== DONE == {final}")
    return final


# ---------------------------------------------------------------- argparse

def _cfg_from_args(args):
    return RunConfig(
        hf_id=getattr(args, "hf_id", None),
        workdir=os.path.abspath(args.workdir),
        preset=getattr(args, "preset", "Q2_K"),
        revision=getattr(args, "revision", None),
        out=getattr(args, "out", None),
        calib=getattr(args, "calib", None),
        llama_cpp=getattr(args, "llama_cpp", None),
        device=getattr(args, "device", "cpu"),
        cuda=getattr(args, "cuda", False),
        cuda_arch=getattr(args, "cuda_arch", None),
        hessian_shards=getattr(args, "hessian_shards", 1),
        n_calib_chunks=getattr(args, "calib_chunks", 192),
        rows_sample=getattr(args, "rows_sample", 16384),
        distill_norm=getattr(args, "norm", False) or getattr(args, "_default_distill", False),
        distill_scales=getattr(args, "scales", False),
        perm=not getattr(args, "no_perm", False),
        acknowledge_unreviewed=getattr(args, "acknowledge_unreviewed", False),
        threads=getattr(args, "threads", 4),
    )


def _add_common(p, hf=True):
    if hf:
        p.add_argument("hf_id")
        p.add_argument("--revision", default=None)
    p.add_argument("--workdir", default="coldpress-work")
    p.add_argument("--preset", default="Q2_K")
    p.add_argument("--calib", default=None, help="calibration text (default: bundled)")
    p.add_argument("--llama-cpp", dest="llama_cpp", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--cuda-arch", dest="cuda_arch", default=None)
    p.add_argument("--hessian-shards", dest="hessian_shards", type=int, default=1)
    p.add_argument("--calib-chunks", dest="calib_chunks", type=int, default=192)
    p.add_argument("--rows-sample", dest="rows_sample", type=int, default=16384)
    p.add_argument("--no-perm", dest="no_perm", action="store_true")
    p.add_argument("--acknowledge-unreviewed", dest="acknowledge_unreviewed",
                   action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    # encode / distill / verify knobs (harmless defaults on every subcommand)
    p.add_argument("--unsafe-storage-order", dest="unsafe_storage_order",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--norm-steps", dest="norm_steps", type=int, default=200)
    p.add_argument("--scale-steps", dest="scale_steps", type=int, default=300)


def build_parser():
    ap = argparse.ArgumentParser("coldpress", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"coldpress {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("onboard"); _add_common(p)
    p = sub.add_parser("calibrate"); _add_common(p)
    p = sub.add_parser("perm"); _add_common(p)
    p = sub.add_parser("encode"); _add_common(p)
    p.add_argument("--norm", action="store_true")
    p.add_argument("--scales", action="store_true")
    p = sub.add_parser("distill"); _add_common(p)
    p.add_argument("--norm", action="store_true")
    p.add_argument("--scales", action="store_true")
    p = sub.add_parser("verify"); _add_common(p, hf=False)
    p.add_argument("--gguf", required=True)
    p.add_argument("--stock", default=None)
    p.add_argument("--ppl-check", dest="ppl_check", default=None)
    p.add_argument("--prompt", default="The capital of France is")
    p = sub.add_parser("quantize"); _add_common(p)
    p.add_argument("--out", default=None)
    p.add_argument("--norm", action="store_true", default=True)
    p.add_argument("--scales", action="store_true")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--ppl-check", dest="ppl_check", default=None)
    p.add_argument("--stock", default=None)
    p = sub.add_parser("build-llamacpp");
    p.add_argument("--dest", default="llama.cpp-build")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--cuda-arch", dest="cuda_arch", default=None)
    p.add_argument("--workdir", default="coldpress-work")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    # verify/build subcommands don't take hf_id
    if args.cmd == "verify":
        args.hf_id = None
    if args.cmd == "build-llamacpp":
        cfg = RunConfig(hf_id=None, workdir=os.path.abspath(args.workdir),
                        cuda=args.cuda, cuda_arch=args.cuda_arch)
        cmd_build_llamacpp(cfg, args)
        return 0
    cfg = _cfg_from_args(args)
    dispatch = {
        "onboard": cmd_onboard, "calibrate": cmd_calibrate, "perm": cmd_perm,
        "encode": lambda c, a: cmd_encode(c, a), "distill": lambda c, a: cmd_distill(c, a),
        "verify": cmd_verify, "quantize": cmd_quantize,
    }
    dispatch[args.cmd](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
