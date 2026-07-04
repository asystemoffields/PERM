#!/usr/bin/env python3
"""Teacher top-K logit cache for distillation (NORM / E3B).

One-time pass: cache the FP teacher's top-K log-probs (+ a single tail-mass bucket) for N
calibration chunks to disk, so training never needs the teacher in RAM. Schema per chunk
npz: ids [CTX] int, top_i [CTX,K] int32, top_lp [CTX,K] f16, tail [CTX] f16.
"""
import os

import numpy as np


def calib_chunks(tokenizer, calib_path, ctx=512, n_chunks=192):
    ids = tokenizer(open(calib_path, encoding="utf-8").read(),
                    return_tensors="pt").input_ids[0]
    n_chunks = min(n_chunks, len(ids) // ctx)
    return [ids[c * ctx:(c + 1) * ctx] for c in range(n_chunks)]


def cache_teacher(model, tokenizer, outdir, calib_path,
                  n_chunks=192, ctx=512, topk=256, log=print):
    import torch
    os.makedirs(outdir, exist_ok=True)
    chunks = calib_chunks(tokenizer, calib_path, ctx, n_chunks)
    model.eval()
    for c, ids in enumerate(chunks):
        f = os.path.join(outdir, f"chunk{c:04d}.npz")
        if os.path.exists(f):
            continue
        with torch.inference_mode():
            lg = model(ids.unsqueeze(0)).logits[0].float()  # [CTX, V]
            lp = torch.log_softmax(lg, -1)
            top_lp, top_i = lp.topk(topk, -1)
            tail = torch.log1p(-top_lp.exp().sum(-1).clamp(max=1 - 1e-7))
        np.savez(f, ids=ids.numpy(), top_i=top_i.numpy().astype(np.int32),
                 top_lp=top_lp.numpy().astype(np.float16),
                 tail=tail.numpy().astype(np.float16))
        if (c + 1) % 16 == 0:
            log(f"teacher chunk {c+1}/{len(chunks)}")
    log(f"teacher targets cached: {len(chunks)} chunks -> {outdir}")
    return outdir
