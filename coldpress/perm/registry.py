#!/usr/bin/env python3
"""Spacemap registry keyed by config.model_type.

get_spacemap(model_type) -> the spacemap module, or None if unknown (PERM is then skipped
with a friendly message; EF/NORM still apply on any arch)."""
import importlib

# model_type (HF config.model_type) -> spacemap module name under .spacemaps
_REGISTRY = {
    "qwen3": "qwen3",
    "qwen35": "qwen35",
    "qwen3_5": "qwen35",
    "gemma4": "gemma4",
    "gemma3": "gemma4",  # provisional: same text-stack family (UNREVIEWED map, gated)
}


def get_spacemap(model_type):
    mod = _REGISTRY.get(model_type)
    if mod is None:
        return None
    return importlib.import_module(f"{__package__}.spacemaps.{mod}")


def available():
    return sorted(set(_REGISTRY))
