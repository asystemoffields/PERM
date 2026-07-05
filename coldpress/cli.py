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
from .config import RunConfig, Workdir, Manifest, text_config
from .perm import registry


def _log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- onboard

# candidate matmul ne[0] values that MUST be divisible by 256 for k-quants
def onboard_report(config):
    # dims come from the TEXT stack (multimodal wrappers nest them under text_config);
    # model_type / spacemap keying stays on the TOP-LEVEL config.
    txt = text_config(config)

    def g(*names, default=None):
        for n in names:
            v = getattr(txt, n, None)
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
        "tie_word_embeddings": bool(getattr(txt, "tie_word_embeddings",
                                            getattr(config, "tie_word_embeddings", False))),
    }


def cmd_onboard(cfg, args):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(cfg.hf_id, revision=cfg.revision,
                                        trust_remote_code=args.trust_remote_code)
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

    # 4. Hessians (generic collector, sharded). --hessian-layer-range LO:HI collects only
    #    that block subrange (split-kernel shard); the manifest entry + marker namespace the
    #    range so shards do not clobber one another.
    from . import hessians as hess
    from .ggufio import read_typemap
    from transformers import AutoConfig
    hdir = wd.dir("hessians")
    lr = cfg.hessian_layer_range
    stage = f"hessians_{lr[0]}_{lr[1]}" if lr else "hessians"
    marker = os.path.join(hdir, f".done_{lr[0]}_{lr[1]}" if lr else ".done")
    ins = {"f16": f16, "calib": calib, "shards": cfg.hessian_shards,
           "chunks": cfg.n_calib_chunks, "range": f"{lr[0]}:{lr[1]}" if lr else "full"}
    if not mf.is_current(stage, ins, [marker]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tm = read_typemap(f16)
        gguf_ne0 = {n: v["shape"][0] for n, v in tm.items()}
        arch = _arch_of(f16)
        tok = AutoTokenizer.from_pretrained(model_dir)
        ids = tok(open(calib, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
        ctx = 512
        n = min(cfg.n_calib_chunks, len(ids) // ctx)
        batches = [ids[c * ctx:(c + 1) * ctx] for c in range(n)]
        config = AutoConfig.from_pretrained(model_dir)
        n_layers = text_config(config).num_hidden_layers   # wrapper-safe (one helper, everywhere)
        dtype, device_map = _resolve_teacher_dtype(config, cfg)
        _log(f"hessian teacher: dtype={dtype}, device_map={device_map}")
        shards = [lr] if lr else _shard_ranges(n_layers, cfg.hessian_shards)
        for lo, hi in shards:
            kw = {"dtype": dtype}
            if device_map is not None:
                kw["device_map"] = device_map
            model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
            model.eval()
            # accumulators stay on CPU (see collect_hessians); a GPU-resident bf16 teacher
            # forms X^T X on-device and moves it to the host accumulator.
            hs, unmapped = hess.collect_hessians(model, batches, gguf_ne0, arch, n_layers,
                                                 layer_lo=lo, layer_hi=hi, log=_log)
            hess.save_hessians(hs, hdir)
            del model
        open(marker, "w").write("ok")
        mf.record(stage, ins, [marker])
    _log(f"hessians: {hdir}")

    # 5. teacher top-K cache (skippable: it is consumed ONLY by cmd_distill)
    from . import teacher as tea
    tdir = wd.dir("teacher")
    if cfg.no_teacher:
        _log("teacher cache: SKIPPED (--no-teacher; permef-only, no distill)")
    else:
        ins = {"model": os.path.join(model_dir, ".downloaded"), "calib": calib,
               "chunks": cfg.n_calib_chunks}
        marker = os.path.join(tdir, ".done")
        if not mf.is_current("teacher", ins, [marker]):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig as _AC
            tok = AutoTokenizer.from_pretrained(model_dir)
            dtype, device_map = _resolve_teacher_dtype(_AC.from_pretrained(model_dir), cfg)
            kw = {"dtype": dtype}
            if device_map is not None:
                kw["device_map"] = device_map
            _log(f"teacher cache: dtype={dtype}, device_map={device_map}")
            model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
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


def _parse_layer_range(s):
    """'LO:HI' -> (int, int), or None. HI is exclusive."""
    if not s:
        return None
    lo, hi = s.split(":")
    return (int(lo), int(hi))


def _param_count_estimate(config):
    """Rough total-parameter estimate from a text config (embeddings + per-layer attn+ffn).
    Used only to pick the default teacher dtype (float32 < 3B, bfloat16 at/above)."""
    txt = text_config(config)

    def g(*names, default=0):
        for n in names:
            v = getattr(txt, n, None)
            if v is not None:
                return v
        return default
    d = g("hidden_size")
    ffn = g("intermediate_size")
    nl = g("num_hidden_layers")
    vocab = g("vocab_size")
    nh = g("num_attention_heads")
    nkv = g("num_key_value_heads", default=nh)
    hd = g("head_dim", default=(d // nh if d and nh else 0))
    if not (d and ffn and nl):
        return 0
    per_attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d          # q,k,v,o (approx)
    per_ffn = 3 * d * ffn                                            # gate,up,down
    return int(2 * vocab * d + nl * (per_attn + per_ffn))            # embd (+tied lm_head)


def _resolve_teacher_dtype(config, cfg):
    """Return (torch.dtype, device_map|None). dtype: explicit cfg.teacher_dtype, else auto by
    param count (bfloat16 at/above 3B). device_map: 'auto' when teacher_device is auto/cuda and
    CUDA is present, else None (CPU)."""
    import torch
    if cfg.teacher_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif cfg.teacher_dtype == "float32":
        dtype = torch.float32
    else:
        n = _param_count_estimate(config)
        dtype = torch.bfloat16 if n >= 3_000_000_000 else torch.float32
    device_map = None
    if cfg.teacher_device in ("cuda", "auto"):
        if torch.cuda.is_available():
            device_map = "auto"
        elif cfg.teacher_device == "cuda":
            raise SystemExit("--teacher-device cuda requested but CUDA is not available")
    return dtype, device_map


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
                                          rows_sample=cfg.rows_sample, log=_log, **ack)
        sm.save_perms(perms, perms_npz, dims, **ack)
        mf.record("perm_optimize", ins, [perms_npz], extra=report)
    perms = sm.load_perms(perms_npz, **ack)

    # G3 gate on the real model. For big models the fp32 load OOMs, so reuse the teacher-dtype
    # logic (bf16 at/above 3B); a bf16 forward is not bit-exact, so the gate threshold is
    # loosened to 5e-3 (documented) -- the permutation is still exact index moves, only the
    # forward arithmetic differs by bf16 rounding.
    tok_ids = _calib_ids(cal["model_dir"], cfg.resolved_calib(), 256)
    g_dtype, g_device_map = _resolve_teacher_dtype(config, cfg)
    threshold = 5e-3 if g_dtype == torch.bfloat16 else 1e-4
    kw = {"dtype": g_dtype}
    if g_device_map is not None:
        kw["device_map"] = g_device_map
    model = AutoModelForCausalLM.from_pretrained(cal["model_dir"], **kw)
    tok_ids = tok_ids.to(next(model.parameters()).device)
    d, rel = gates.g3(sm, model, perms, dims, tok_ids, threshold=threshold, **ack)
    _log(f"[perm] G3 PASS: rel logits drift {rel:.3e} (max|dlogit|={d:.3e}) "
         f"[dtype={g_dtype}, threshold={threshold:.0e}]")
    del model

    # --perms-only: stop after the gate. The split scale chain re-derives the permuted
    # checkpoint / permuted f16 / permuted imatrix DETERMINISTICALLY from perms.npz in a
    # downstream kernel (apply_perms is pure index_select), so the big artifacts never cross
    # kernel boundaries. This also sidesteps _write_permuted_checkpoint's single-file
    # model.safetensors assumption, which does not hold for sharded big-model snapshots.
    if getattr(args, "perms_only", False):
        _log(f"[perm] --perms-only: perms.npz saved + G3 gate passed; skipping permuted "
             f"checkpoint / f16 / imatrix materialization (downstream re-derives them from "
             f"{perms_npz})")
        return {"spacemap": sm, "perms": perms, "perms_npz": perms_npz, "dims": dims,
                "f16": None, "imatrix": None, "model_dir": None,
                "acknowledge_unreviewed": bool(ack)}

    # apply perms to a copy of the checkpoint, convert -> f16
    pdir = wd.dir("perm")
    permuted_model = os.path.join(pdir, "model-permuted")
    _write_permuted_checkpoint(sm, cal["model_dir"], permuted_model, perms, dims, ack,
                               strip_vision=cfg.strip_vision)
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
        perm_im.permute_imatrix(sm, cal["imatrix"], perm_imat, perms, dims, log=_log, **ack)
        mf.record("perm_imatrix", ins, [perm_imat])
    return {"spacemap": sm, "perms": perms, "perms_npz": perms_npz, "dims": dims,
            "f16": perm_f16, "imatrix": perm_imat, "imatrix_orig": cal["imatrix"],
            "model_dir": permuted_model, "acknowledge_unreviewed": bool(ack)}


def _calib_ids(model_dir, calib, n_tok):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = open(calib, encoding="utf-8").read()[:20000]
    return tok(text, return_tensors="pt").input_ids[:, :n_tok]


def _write_permuted_checkpoint(sm, src_dir, out_dir, perms, dims, ack, strip_vision=False):
    import shutil
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    # load state dict (single-file safetensors expected for small models)
    st = os.path.join(src_dir, "model.safetensors")
    sd = load_file(st)
    # strip_vision is honoured by the gemma4 spacemap (text-only ship: delete vision-tower /
    # multi_modal_projector tensors); other spacemaps accept and ignore it (no-op).
    out = sm.apply_perms(sd, perms, dims, consume=True, strip_vision=strip_vision, **ack)
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
        f16, imatrix = source["f16"], source["imatrix"]   # imatrix = PERMUTED (stock byte-map)
        sm, perms, dims = source["spacemap"], source["perms"], source["dims"]
        # encode_gguf permutes the ORIGINAL-order imatrix itself (qw[p], matching the validated
        # research e2_ef.py); handing it the ALREADY-permuted imatrix double-permutes the weights
        # (measured on 0.6B: median err-vs-stock +188%, wiki +58%). So the stock byte-map target
        # uses the permuted imatrix, but the EF weighting uses the original.
        imatrix_ef = source.get("imatrix_orig", imatrix)
        enc_ack = bool(source.get("acknowledge_unreviewed"))
        tag = "permef"
    else:
        f16 = os.path.join(wd.dir("f16"), "model-f16.gguf")
        imatrix = os.path.join(wd.dir("imatrix"), "imatrix.gguf")
        imatrix_ef = imatrix    # stock orientation: no perms, encode_gguf leaves it as-is
        sm = perms = dims = None
        enc_ack = False
        tag = "ef"
    if imatrix and not os.path.exists(imatrix):
        imatrix = None  # calibrate builds it; without it, quantize/EF use the ref path
    if imatrix_ef and not os.path.exists(imatrix_ef):
        imatrix_ef = None

    # stock quant at preset (the byte-map target)
    stock = os.path.join(wd.dir("encode"), f"stock-{tag}-{cfg.preset}.gguf")
    ins = {"f16": f16, "imatrix": imatrix or "", "preset": cfg.preset}
    if not mf.is_current(f"stock_{tag}", ins, [stock]):
        llama.quantize(f16, stock, cfg.preset, imatrix=imatrix,
                       logfile=wd.log(f"stock-{tag}.log"))
        mf.record(f"stock_{tag}", ins, [stock], llama_commit=llamacpp.PINNED_COMMIT)

    out = os.path.join(wd.dir("encode"), f"coldpress-{tag}-{cfg.preset}.gguf")
    ins = {"f16": f16, "stock": stock, "hessians": os.path.join(wd.dir("hessians"), ".done"),
           "imatrix": imatrix_ef, "unsafe": args.unsafe_storage_order}
    if not mf.is_current(f"encode_{tag}", ins, [out]):
        ef.encode_gguf(f16, stock, wd.dir("hessians"), out, imatrix_path=imatrix_ef,
                       perms=perms, spacemap=sm, dims=dims, acknowledge_unreviewed=enc_ack,
                       unsafe_storage_order=args.unsafe_storage_order,
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
    n_layers = text_config(config).num_hidden_layers   # wrapper-safe (one helper, everywhere)
    teacher_dir = wd.dir("teacher")
    cur = enc["gguf"]
    # bf16 skeleton for big models (same teacher-dtype rule); trainable d/dmin/norm stay f32.
    dtype, _dm = _resolve_teacher_dtype(config, cfg)
    _log(f"[distill] student skeleton dtype={dtype}, device={cfg.device}")

    if cfg.distill_norm:
        out = os.path.join(wd.dir("distill"), os.path.basename(cur).replace(".gguf", "-norm.gguf"))
        ins = {"in": cur, "teacher": os.path.join(teacher_dir, ".done"), "steps": args.norm_steps}
        if not mf.is_current("distill_norm", ins, [out]):
            distill.distill_norm(model_dir, arch, n_layers, cur, teacher_dir, out,
                                 steps=args.norm_steps, dtype=dtype, device=cfg.device, log=_log)
            mf.record("distill_norm", ins, [out])
        cur = out
    if cfg.distill_scales:
        out = os.path.join(wd.dir("distill"), os.path.basename(cur).replace(".gguf", "-e3b.gguf"))
        ins = {"in": cur, "teacher": os.path.join(teacher_dir, ".done"), "steps": args.scale_steps}
        if not mf.is_current("distill_scales", ins, [out]):
            distill.distill_e3b(model_dir, arch, n_layers, cur, teacher_dir, out,
                                steps=args.scale_steps, device=cfg.device, dtype=dtype, log=_log)
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
        hessian_layer_range=_parse_layer_range(getattr(args, "hessian_layer_range", None)),
        n_calib_chunks=getattr(args, "calib_chunks", 192),
        rows_sample=getattr(args, "rows_sample", 16384),
        teacher_dtype=getattr(args, "teacher_dtype", None),
        teacher_device=getattr(args, "teacher_device", "cpu"),
        no_teacher=getattr(args, "no_teacher", False),
        distill_norm=getattr(args, "norm", False) or getattr(args, "_default_distill", False),
        distill_scales=getattr(args, "scales", False),
        perm=not getattr(args, "no_perm", False),
        acknowledge_unreviewed=getattr(args, "acknowledge_unreviewed", False),
        strip_vision=getattr(args, "strip_vision", False),
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
    p.add_argument("--hessian-layer-range", dest="hessian_layer_range", default=None,
                   help="LO:HI -- collect Hessians only for blocks [LO,HI) (split-kernel "
                        "shard, e.g. the 12B two-shard plan). Manifest entry is namespaced "
                        "per range.")
    p.add_argument("--teacher-dtype", dest="teacher_dtype", default=None,
                   choices=["float32", "bfloat16"],
                   help="dtype for the FP teacher/skeleton load (default: float32 below 3B "
                        "params, bfloat16 at/above -- decided from the config).")
    p.add_argument("--teacher-device", dest="teacher_device", default="cpu",
                   choices=["cpu", "cuda", "auto"],
                   help="device for the FP teacher/skeleton (auto/cuda -> device_map='auto' "
                        "when CUDA is present; Hessian accumulators stay on CPU).")
    p.add_argument("--no-teacher", dest="no_teacher", action="store_true",
                   help="skip the teacher top-K cache (calibrate step 5). For permef-only "
                        "workflows that never distill; do NOT combine with quantize's default "
                        "--norm distill, which requires the teacher.")
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
    p.add_argument("--strip-vision", dest="strip_vision", action="store_true",
                   help="gemma4: drop vision-tower/projector tensors before permuting "
                        "(text-only ship); no-op on other spacemaps")
    p.add_argument("--perms-only", dest="perms_only", action="store_true",
                   help="stop after the G3 gate: save perms.npz but skip materializing the "
                        "permuted checkpoint / f16 / imatrix (split scale chain: a downstream "
                        "kernel re-derives them deterministically from perms.npz)")
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
    p.add_argument("--strip-vision", dest="strip_vision", action="store_true",
                   help="gemma4: drop vision-tower/projector tensors before permuting "
                        "(text-only ship); no-op on other spacemaps")
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
