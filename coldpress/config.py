#!/usr/bin/env python3
"""Run configuration, workdir layout, and the resumability/provenance manifest.

Every stage records a manifest entry (input fingerprints, tool versions, timestamps,
outputs). A stage is skipped when its recorded inputs still match and its outputs exist --
so `coldpress quantize` is resumable from a --workdir cache and each artifact carries a
provenance trail.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict


def text_config(config):
    """Unwrap a multimodal wrapper config to its text stack.

    Real big-model repos (Qwen3.5-*, gemma-4-*) ship any-to-any wrapper configs whose text
    dims (num_hidden_layers, hidden_size, ...) nest under `text_config`; plain text configs
    are returned unchanged. Every dims/num_hidden_layers read in the package goes through
    this ONE helper -- keep it that way (the spacemaps unwrap internally with the same rule).
    NOTE: config.model_type stays a TOP-LEVEL read everywhere (the spacemap registry keys on
    the wrapper's model_type, e.g. 'gemma4_unified')."""
    tc = getattr(config, "text_config", None)
    return tc if tc is not None else config


def _versions(llama_commit=None):
    v = {}
    try:
        import importlib.metadata as m
        for pkg in ("numpy", "torch", "transformers", "gguf", "safetensors"):
            try:
                v[pkg] = m.version(pkg)
            except Exception:
                pass
    except Exception:
        pass
    from . import __version__
    v["coldpress"] = __version__
    if llama_commit:
        v["llama.cpp"] = llama_commit
    return v


def file_fingerprint(path):
    """Cheap, stable fingerprint of a file input: (size, mtime_ns). Directories are walked."""
    if not os.path.exists(path):
        return None
    if os.path.isdir(path):
        acc = []
        for root, _dirs, files in os.walk(path):
            for f in sorted(files):
                p = os.path.join(root, f)
                st = os.stat(p)
                acc.append((os.path.relpath(p, path), st.st_size, st.st_mtime_ns))
        return {"dir": sorted(acc)}
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def content_hash(path, limit=1 << 20):
    """sha256 over the whole file (small) or first+last `limit` bytes (large)."""
    if not os.path.exists(path) or os.path.isdir(path):
        return None
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size <= 2 * limit:
            h.update(f.read())
        else:
            h.update(f.read(limit))
            f.seek(-limit, os.SEEK_END)
            h.update(f.read(limit))
    return h.hexdigest()


@dataclass
class RunConfig:
    hf_id: str
    workdir: str
    preset: str = "Q2_K"
    revision: str = None
    out: str = None
    calib: str = None            # None -> bundled data/fineweb-calib.txt
    llama_cpp: str = None        # explicit llama.cpp dir (else discovery)
    device: str = "cpu"
    cuda: bool = False
    cuda_arch: str = None
    hessian_shards: int = 1
    hessian_layer_range: tuple = None   # (lo, hi) subrange override for a split-kernel shard
    n_calib_chunks: int = 192
    rows_sample: int = 16384
    teacher_dtype: str = None     # None -> auto (float32 < 3B params, bfloat16 at/above)
    teacher_device: str = "cpu"   # cpu | cuda | auto  (auto uses cuda + device_map when avail)
    distill_norm: bool = True
    distill_scales: bool = False
    perm: bool = True            # attempt PERM if a spacemap exists
    acknowledge_unreviewed: bool = False
    strip_vision: bool = False   # gemma4: drop vision-tower/projector tensors, text-only ship
    threads: int = 4

    def resolved_calib(self):
        if self.calib:
            return os.path.abspath(self.calib)
        return os.path.join(os.path.dirname(__file__), "data", "fineweb-calib.txt")


class Workdir:
    """Directory layout under a run's --workdir. Creates dirs lazily."""

    SUBDIRS = ["model", "f16", "imatrix", "hessians", "teacher", "perm", "encode",
               "distill", "logs"]

    def __init__(self, root):
        self.root = os.path.abspath(root)

    def path(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p

    def dir(self, name):
        p = os.path.join(self.root, name)
        os.makedirs(p, exist_ok=True)
        return p

    def log(self, name):
        return os.path.join(self.dir("logs"), name)


class Manifest:
    """workdir/manifest.json: {stage: {inputs, versions, outputs, timestamp}}."""

    def __init__(self, workdir):
        self.path = os.path.join(workdir.root if isinstance(workdir, Workdir) else workdir,
                                 "manifest.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = {}
        if os.path.exists(self.path):
            try:
                self.data = json.load(open(self.path))
            except Exception:
                self.data = {}

    @staticmethod
    def _fingerprint_inputs(inputs):
        """inputs: {key: value}. File-path values (str existing paths) are fingerprinted;
        other values are used verbatim."""
        fp = {}
        for k, v in inputs.items():
            if isinstance(v, str) and os.path.exists(v):
                fp[k] = file_fingerprint(v)
            else:
                fp[k] = v
        return fp

    def _hash(self, inputs):
        fp = self._fingerprint_inputs(inputs)
        blob = json.dumps(fp, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def is_current(self, stage, inputs, outputs):
        e = self.data.get(stage)
        if not e:
            return False
        if e.get("inputs_hash") != self._hash(inputs):
            return False
        return all(os.path.exists(o) for o in outputs)

    def record(self, stage, inputs, outputs, extra=None, llama_commit=None):
        self.data[stage] = {
            "inputs_hash": self._hash(inputs),
            "inputs": self._fingerprint_inputs(inputs),
            "outputs": list(outputs),
            "versions": _versions(llama_commit),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "extra": extra or {},
        }
        self.save()

    def save(self):
        json.dump(self.data, open(self.path, "w"), indent=1, default=str)
