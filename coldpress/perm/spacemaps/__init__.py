"""Spacemap registry package.

A spacemap declares every permutable internal space of an architecture and the exact,
value-free tensor edits that realize a coordinated permutation. Each module exposes:

    MODEL_TYPE, ARCH                         config.model_type and GGUF arch strings
    dims_from_config(config) -> dict         d_model,d_ffn,n_layers,n_heads,n_kv,head_dim
    identity_perms(dims)                     the identity element
    save_perms(perms,path) / load_perms(path)
    apply_perms(state_dict, perms, dims, consume=False, strip_vision=False)
                                            pure index_select on an HF sd; strip_vision drops
                                            vision-tower/projector tensors (gemma4), else no-op
    apply_perms_inplace(model, perms, dims)               in-place on a loaded HF model
    input_perm(gguf_tensor_name, perms) -> perm|None      ne[0]-axis perm (imatrix/Hessian)
    optimize(weights, ttypes, qws, dims, rows_sample) -> (perms, report)
    g3_check(model, perms, dims, ids) -> (max|dlogit|, rel)   logits-equality oracle
    permute_imatrix(src, dst, perms, dims)   permute an imatrix GGUF to match

qwen35 and gemma4 ship as Fable derivations pending review + a G3 gate on real weights;
they raise NotImplementedError unless called with acknowledge_unreviewed=True.
"""
