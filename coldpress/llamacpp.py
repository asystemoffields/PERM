#!/usr/bin/env python3
"""Locate, pin-build, and drive llama.cpp.

All quantization + eval must run against the pinned commit so encoder-faithfulness and ppl
numbers are reproducible. No pipe-masked steps: every build/run writes its output to a log
file and then ASSERTS the expected binaries/artifacts exist (a pipe would let a failed
build return 0 and be discovered only at a cryptic 'exit 127' later).
"""
import os
import shutil
import subprocess

PINNED_COMMIT = "039e20a2db9e87b2477c76cc04905f3e1acad77f"
REPO_URL = "https://github.com/ggml-org/llama.cpp"
BINARIES = ["llama-quantize", "llama-perplexity", "llama-cli", "llama-imatrix"]

# known-good CUDA flag set (RECIPE.md landmine table 3): GGML_CUDA_NO_VMM=ON is required
# or the CUDA cmake fails in seconds against the driver stub link.
_CUDA_FLAGS = ["-DGGML_CUDA=ON", "-DGGML_CUDA_NO_VMM=ON"]


class LlamaCpp:
    def __init__(self, bin_dir, repo_dir=None):
        self.bin_dir = bin_dir
        self.repo_dir = repo_dir

    def path(self, name):
        p = os.path.join(self.bin_dir, name)
        assert os.path.exists(p), f"binary not found: {p}"
        return p

    def has(self, name):
        return os.path.exists(os.path.join(self.bin_dir, name))

    # ---- subprocess wrappers (never pipe-masked; asserts on outputs) ----

    def _run(self, argv, logfile=None, cwd=None):
        if logfile:
            with open(logfile, "w") as lf:
                r = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd)
        else:
            r = subprocess.run(argv, cwd=cwd)
        if r.returncode != 0:
            tail = ""
            if logfile and os.path.exists(logfile):
                tail = "".join(open(logfile, errors="replace").readlines()[-25:])
            raise RuntimeError(f"command failed ({r.returncode}): {' '.join(argv)}\n{tail}")
        return r

    def quantize(self, f16_gguf, out_gguf, preset, imatrix=None, logfile=None, extra=None):
        argv = [self.path("llama-quantize")]
        if imatrix:
            argv += ["--imatrix", imatrix]
        argv += list(extra or [])
        argv += [f16_gguf, out_gguf, preset]
        self._run(argv, logfile)
        assert os.path.exists(out_gguf) and os.path.getsize(out_gguf) > 0, \
            f"llama-quantize produced no output: {out_gguf}"
        return out_gguf

    def imatrix(self, model_gguf, calib_txt, out_imatrix, ctx=512, ngl=0, logfile=None,
                extra=None):
        argv = [self.path("llama-imatrix"), "-m", model_gguf, "-f", calib_txt,
                "-o", out_imatrix, "-c", str(ctx), "-ngl", str(ngl)]
        argv += list(extra or [])
        self._run(argv, logfile)
        assert os.path.exists(out_imatrix) and os.path.getsize(out_imatrix) > 0, \
            f"llama-imatrix produced no output: {out_imatrix}"
        return out_imatrix

    def perplexity(self, model_gguf, corpus_txt, ctx=512, ngl=0, threads=4, logfile=None,
                   extra=None):
        argv = [self.path("llama-perplexity"), "-m", model_gguf, "-f", corpus_txt,
                "-c", str(ctx), "-ngl", str(ngl), "-t", str(threads)]
        argv += list(extra or [])
        self._run(argv, logfile)
        return logfile

    def cli(self, model_gguf, prompt, n_predict=48, ngl=0, logfile=None, extra=None):
        argv = [self.path("llama-cli"), "-m", model_gguf, "-p", prompt,
                "-n", str(n_predict), "-ngl", str(ngl), "-no-cnv"]
        argv += list(extra or [])
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = r.stdout.decode(errors="replace")
        if logfile:
            open(logfile, "w").write(out)
        if r.returncode != 0:
            raise RuntimeError(f"llama-cli failed ({r.returncode}); tail:\n{out[-800:]}")
        return out

    def convert_hf_to_gguf(self, hf_dir, out_gguf, outtype="f16", logfile=None, extra=None):
        assert self.repo_dir, ("convert_hf_to_gguf.py requires the llama.cpp source tree; "
                               "run `coldpress build-llamacpp` or pass --llama-cpp a repo dir")
        script = os.path.join(self.repo_dir, "convert_hf_to_gguf.py")
        assert os.path.exists(script), f"convert_hf_to_gguf.py not found in {self.repo_dir}"
        import sys
        argv = [sys.executable, script, hf_dir, "--outfile", out_gguf, "--outtype", outtype]
        argv += list(extra or [])
        self._run(argv, logfile)
        assert os.path.exists(out_gguf) and os.path.getsize(out_gguf) > 0, \
            f"convert produced no output: {out_gguf}"
        return out_gguf


# ---------------------------------------------------------------- discovery

def _bin_dir_of(root):
    """Given a llama.cpp dir, find the directory holding the built binaries."""
    for cand in [os.path.join(root, "build", "bin"), os.path.join(root, "bin"), root]:
        if os.path.exists(os.path.join(cand, "llama-quantize")):
            return cand
    return None


def locate(explicit_dir=None):
    """Find an existing llama.cpp. Order: explicit_dir -> $COLDPRESS_LLAMACPP -> PATH.
    Returns LlamaCpp or None."""
    for root in [explicit_dir, os.environ.get("COLDPRESS_LLAMACPP")]:
        if root and os.path.isdir(root):
            bd = _bin_dir_of(root)
            if bd:
                # repo_dir is the tree containing convert_hf_to_gguf.py, if present
                repo = root if os.path.exists(os.path.join(root, "convert_hf_to_gguf.py")) else None
                if repo is None:
                    parent = os.path.dirname(os.path.dirname(bd))
                    if os.path.exists(os.path.join(parent, "convert_hf_to_gguf.py")):
                        repo = parent
                return LlamaCpp(bd, repo)
    onpath = shutil.which("llama-quantize")
    if onpath:
        return LlamaCpp(os.path.dirname(onpath), None)
    return None


def build(dest, cuda=False, cuda_arch=None, commit=PINNED_COMMIT, jobs=None, log=print):
    """Clone the pinned commit into `dest` and cmake-build the required binaries.

    CPU by default; cuda=True adds the known-good flag set. Returns a LlamaCpp. Every step
    asserts its outputs -- no pipe masking."""
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    logdir = os.path.join(dest, "_build_logs")
    os.makedirs(logdir, exist_ok=True)

    def run(argv, logname, cwd=None):
        log(f"[build-llamacpp] {' '.join(argv)}")
        with open(os.path.join(logdir, logname), "w") as lf:
            r = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd)
        if r.returncode != 0:
            tail = "".join(open(os.path.join(logdir, logname), errors="replace").readlines()[-30:])
            raise RuntimeError(f"build step failed ({r.returncode}): {' '.join(argv)}\n{tail}")

    src = os.path.join(dest, "llama.cpp")
    if not os.path.exists(os.path.join(src, ".git")):
        run(["git", "clone", REPO_URL, src], "clone.log")
    run(["git", "-C", src, "fetch", "--depth", "1", "origin", commit], "fetch.log")
    run(["git", "-C", src, "checkout", commit], "checkout.log")

    build_dir = os.path.join(src, "build")
    cfg = ["cmake", "-S", src, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release",
           "-DLLAMA_CURL=OFF"]
    if cuda:
        cfg += _CUDA_FLAGS
        cfg.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}" if cuda_arch
                   else "-DCMAKE_CUDA_ARCHITECTURES=native")
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if os.path.exists(nvcc):
            cfg.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    run(cfg, "cmake_configure.log")

    jobs = jobs or (os.cpu_count() or 4)
    run(["cmake", "--build", build_dir, "--config", "Release", "-j", str(jobs),
         "--target"] + BINARIES, "cmake_build.log")

    bd = _bin_dir_of(src)
    assert bd, f"build finished but no bin dir with llama-quantize under {src}"
    missing = [b for b in BINARIES if not os.path.exists(os.path.join(bd, b))]
    assert not missing, f"build did not produce: {missing}"
    log(f"[build-llamacpp] built {BINARIES} in {bd}")
    return LlamaCpp(bd, src)
