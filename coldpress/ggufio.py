#!/usr/bin/env python3
"""GGUF re-emission: read a GGUF, write a new one with (possibly replaced) raw tensor bytes.

G0 gate: re-emitting a stock llama.cpp artifact verbatim must yield an artifact that
llama.cpp treats identically (byte-identical tensor data + logically identical metadata).
This is the write path EF uses to ship its own encodings.

gguf API: uses the pip `gguf` package (>=0.19). Notes on the API surface used here,
verified against 0.19.0:
  * GGUFReader(path).tensors -> ReaderTensor(name, tensor_type, shape, n_elements,
    n_bytes, data_offset, data, field). Quantized `.data` is uint8 [nrow, row_bytes];
    `.shape` is in ne order (shape[0] == innermost n_per_row).
  * GGUFReader(path).fields -> ReaderField(offset, name, parts, data, types).
  * GGUFWriter(path, arch).add_key_value(key, val, vtype, sub_type=itype).
  * GGUFWriter.add_tensor(name, ndarray, raw_shape=BYTE_shape, raw_dtype=GGMLType);
    when re-adding a reader's already-quantized uint8 array we omit raw_shape and pass
    raw_dtype -- the writer derives the logical shape from the byte shape.
"""
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType
import numpy as np


def _field_value(f):
    vtype = f.types[0]
    if vtype == GGUFValueType.ARRAY:
        itype = f.types[1]
        if itype == GGUFValueType.STRING:
            return [str(bytes(f.parts[i]), "utf-8") for i in f.data], vtype, itype
        return [f.parts[i].item() for i in f.data], vtype, itype
    if vtype == GGUFValueType.STRING:
        return str(bytes(f.parts[f.data[0]]), "utf-8"), vtype, None
    v = f.parts[f.data[0]].item()
    if vtype == GGUFValueType.BOOL:
        v = bool(v)
    return v, vtype, None


def read_arch(reader):
    for f in reader.fields.values():
        if f.name == "general.architecture":
            return str(bytes(f.parts[f.data[0]]), "utf-8")
    return None


def reemit(src, dst, replace=None, replace_f32=None):
    """Copy src GGUF to dst through our writer.
    replace: {tensor_name: uint8 raw bytes} for quantized tensors;
    replace_f32: {tensor_name: f32 bytes/array} for F32 tensors (norm gains)."""
    r = GGUFReader(src)
    arch = read_arch(r)
    assert arch, "no architecture key"
    w = GGUFWriter(dst, arch)

    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for f in r.fields.values():
        if f.name in skip:
            continue
        val, vtype, itype = _field_value(f)
        w.add_key_value(f.name, val, vtype, sub_type=itype)

    n_replaced = 0
    for t in r.tensors:
        data = np.asarray(t.data)
        if replace and t.name in replace:
            # only for quantized tensors, whose reader data is uint8 [nrow, row_bytes]
            assert data.dtype == np.uint8, f"replace needs a quantized tensor: {t.name}"
            raw = np.asarray(replace[t.name], dtype=np.uint8)
            assert raw.nbytes == t.n_bytes, (t.name, raw.nbytes, t.n_bytes)
            data = raw.reshape(data.shape)
            n_replaced += 1
        if replace_f32 and t.name in replace_f32:
            assert data.dtype == np.float32, f"replace_f32 needs an F32 tensor: {t.name}"
            v = np.frombuffer(np.asarray(replace_f32[t.name]).tobytes()
                              if not isinstance(replace_f32[t.name], bytes)
                              else replace_f32[t.name], dtype=np.float32)
            assert v.nbytes == t.n_bytes, (t.name, v.nbytes, t.n_bytes)
            data = v.reshape(data.shape)
            n_replaced += 1
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"re-emitted {len(r.tensors)} tensors ({n_replaced} replaced) -> {dst}")


def compare(a_path, b_path):
    """Tensor-level byte comparison + KV logical comparison."""
    a, b = GGUFReader(a_path), GGUFReader(b_path)
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    assert set(ta) == set(tb)
    bad = [n for n in ta
           if not np.array_equal(np.asarray(ta[n].data).view(np.uint8),
                                 np.asarray(tb[n].data).view(np.uint8))]
    ka = {f.name: _field_value(f)[0] for f in a.fields.values() if not f.name.startswith("GGUF.")}
    kb = {f.name: _field_value(f)[0] for f in b.fields.values() if not f.name.startswith("GGUF.")}
    kv_bad = [k for k in ka if ka.get(k) != kb.get(k)] + [k for k in kb if k not in ka]
    print(f"tensors: {len(ta)} compared, {len(bad)} data mismatches: {bad[:5]}")
    print(f"kv: {len(ka)} vs {len(kb)}, {len(kv_bad)} mismatches: {kv_bad[:5]}")
    return not bad and not kv_bad


def load_imatrix_means(path):
    """Read an imatrix GGUF into {tensor_name: mean(X^2) f32 [ne0]}.

    An imatrix GGUF stores, per weight tensor, <name>.in_sum2 (F32 sum of squared input
    activations per column) and <name>.counts. The per-column mean is in_sum2/count; that
    is the `qw` weight vector llama.cpp's weighted encoder uses."""
    r = GGUFReader(path)
    sums, counts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-len(".in_sum2")]] = np.array(t.data, dtype=np.float32)
        elif t.name.endswith(".counts"):
            counts[t.name[:-len(".counts")]] = np.array(t.data, dtype=np.float32)
    means = {}
    for k, s in sums.items():
        c = counts[k]
        cc = float(c.reshape(-1)[0])
        means[k] = (s / np.float32(cc)).astype(np.float32)
    return means


def read_typemap(path):
    """Per-tensor {name: {type, shape (ne order), n_bytes}} for the byte-parity gate."""
    r = GGUFReader(path)
    m = {}
    for t in r.tensors:
        m[t.name] = {
            "type": t.tensor_type.name,
            "shape": [int(x) for x in t.shape],
            "n_bytes": int(t.n_bytes),
        }
    return m


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "compare":
        ok = compare(sys.argv[2], sys.argv[3])
        sys.exit(0 if ok else 1)
    reemit(sys.argv[1], sys.argv[2])
