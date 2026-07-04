"""coldpress: container-aware encoding of standard GGUF k-quants.

At a fixed low-bit operating point, produce a GGUF whose held-out perplexity is lower than
the best stock-llama.cpp artifact of equal-or-smaller file size -- while staying a byte-map
-identical standard GGUF that loads in stock llama.cpp with no code changes. Only the
encoder that chooses the bits differs.

Three stages, each independently ablatable:
  PERM   function-preserving channel permutation that regroups k-quant sub-blocks (tier 2,
         needs an architecture spacemap). Zero bytes, zero runtime, exactly reversible.
  EF     GPTQ-style Hessian error feedback encoded into the standard k-quant container
         (tier 1, any llama.cpp arch). act_order always on.
  NORM/E3B  distill the container's free F32 norm gains (and fp16 superblock scales)
         against the FP teacher (tier 1).
"""
__version__ = "0.1.0"

# Pinned llama.cpp commit for all quantization + eval (see llamacpp.PINNED_COMMIT).
LLAMACPP_COMMIT = "039e20a2db9e87b2477c76cc04905f3e1acad77f"
