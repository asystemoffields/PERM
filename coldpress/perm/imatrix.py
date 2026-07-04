#!/usr/bin/env python3
"""Permute an imatrix GGUF to match a PERM-transformed model.

Delegates to the spacemap's permute_imatrix: activation stats move with channels under a
permutation, exactly -- permute each <name>.in_sum2 along ne[0] with the same perm as the
weight's ne[0]. counts unchanged."""


def permute_imatrix(spacemap, src, dst, perms, dims, log=print):
    n = spacemap.permute_imatrix(src, dst, perms, dims)
    log(f"patched {n} in_sum2 tensors -> {dst}")
    return n
