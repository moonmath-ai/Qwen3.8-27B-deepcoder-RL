"""Tinker wire format <-> plain numpy, server side.

Inverse of the pinned SDK's own converters (tinker/proto/request_conv.py and response_conv.py),
using the SDK's generated protobuf module, so the two sides cannot drift within a pinned version.
Spec: the pinned SDK's own types and protobuf module (tinker 0.33.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import zstandard
from tinker.proto import tinker_public_pb2 as pb

_NP = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64, pb.DTYPE_INT32: np.int32}
_PB = {np.dtype(np.float32): pb.DTYPE_FLOAT32, np.dtype(np.int64): pb.DTYPE_INT64,
       np.dtype(np.int32): pb.DTYPE_INT32}


class WireError(ValueError):
    """Malformed or unsupported client input (reported to the client as category=user)."""


@dataclass
class Datum:
    tokens: np.ndarray                      # int64 [N], model input (text chunks concatenated)
    inputs: dict[str, np.ndarray]           # loss_fn_inputs, each [N] (dense, decoded from CSR)


@dataclass
class FwdBwdRequest:
    model_id: str
    seq_id: int
    loss_fn: str
    config: dict[str, float | str]
    forward_only: bool
    data: list[Datum] = field(default_factory=list)


def _tensor(msg: pb.Tensor) -> np.ndarray:
    dt = _NP.get(msg.dtype)
    if dt is None:
        raise WireError(f"unsupported tensor dtype {msg.dtype}")
    shape = list(msg.shape)
    if msg.HasField("sparse_csr"):
        csr = msg.sparse_csr
        vals = np.frombuffer(csr.values, dtype=dt)
        crow = np.frombuffer(csr.crow_indices, dtype=np.int64)
        col = np.frombuffer(csr.col_indices, dtype=np.int64)
        if len(shape) != 2:
            raise WireError("sparse tensor needs a 2-D shape")
        out = np.zeros(shape, dtype=dt)
        for r in range(shape[0]):
            out[r, col[crow[r]:crow[r + 1]]] = vals[crow[r]:crow[r + 1]]
        return out
    arr = np.frombuffer(msg.dense, dtype=dt).copy()
    return arr.reshape(shape) if shape else arr


def decode_forward_backward(body: bytes, content_encoding: str | None) -> FwdBwdRequest:
    if content_encoding == "zstd":
        body = zstandard.ZstdDecompressor().decompress(body)
    msg = pb.ForwardBackwardRequest()
    try:
        msg.ParseFromString(body)
    except Exception as e:  # noqa: BLE001 - protobuf raises DecodeError subclasses
        raise WireError("forward_backward body is not a ForwardBackwardRequest") from e
    # v2 wins when present (SDK dual-writes; strings exist only in v2).
    config: dict[str, float | str] = dict(msg.loss_fn_config)
    for k, v in msg.loss_fn_config_v2.items():
        config[k] = v.text if v.WhichOneof("value") == "text" else v.number
    data = []
    for d in msg.data:
        toks = []
        for chunk in d.model_input:
            kind = chunk.WhichOneof("chunk")
            if kind != "encoded_text":
                raise WireError(f"model input chunk type {kind!r} not supported (text only)")
            toks.append(np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32))
        tokens = np.concatenate(toks).astype(np.int64) if toks else np.zeros(0, np.int64)
        data.append(Datum(tokens=tokens, inputs={k: _tensor(t) for k, t in d.loss_fn_inputs.items()}))
    return FwdBwdRequest(msg.model_id, msg.seq_id, msg.loss_fn, config, msg.forward_only, data)


def _batched(arrays: list[np.ndarray]) -> pb.BatchedTensor:
    """Per-datum 1-D arrays -> BatchedTensor with int64 *byte* offsets."""
    dt = arrays[0].dtype if arrays else np.dtype(np.float32)
    bt = pb.BatchedTensor(dtype=_PB[np.dtype(dt)])
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    parts = []
    for i, a in enumerate(arrays):
        b = np.ascontiguousarray(a, dtype=dt).tobytes()
        parts.append(b)
        offsets[i + 1] = offsets[i] + len(b)
    bt.data = b"".join(parts)
    bt.offsets = offsets.tobytes()
    return bt


def encode_forward_backward_output(outputs: list[dict[str, np.ndarray]], metrics: dict[str, float]) -> bytes:
    """outputs: one dict per datum (e.g. {"logprobs": float32 [N]}). Metric keys must be name:reduction."""
    for k in metrics:
        if k.count(":") != 1:
            raise ValueError(f"metric key {k!r} must be name:reduction")
    msg = pb.ForwardBackwardOutput(loss_fn_output_type="ArrayRecord")
    rec = msg.loss_fn_outputs.add(num_datums=len(outputs))
    for name in (outputs[0].keys() if outputs else ()):
        rec.fields[name].CopyFrom(_batched([o[name].reshape(-1) for o in outputs]))
    for k, v in metrics.items():
        msg.metrics[k] = float(v)
    return msg.SerializeToString()


@dataclass
class Sequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str  # "stop" | "length"


def encode_sample_response(seqs: list[Sequence], prompt_logprobs: list[float | None] | None) -> bytes:
    msg = pb.SampleResponse()
    for s in seqs:
        m = msg.sequences.add()
        m.stop_reason = pb.STOP_REASON_STOP if s.stop_reason == "stop" else pb.STOP_REASON_LENGTH
        m.tokens = np.asarray(s.tokens, dtype=np.int32).tobytes()
        m.logprobs = np.asarray(s.logprobs, dtype=np.float32).tobytes()
    if prompt_logprobs is not None:
        # NaN where not computed; position 0 must be NaN.
        arr = np.array([np.nan if v is None else v for v in prompt_logprobs], dtype=np.float32)
        if len(arr):
            arr[0] = np.nan
        msg.prompt_logprobs = arr.tobytes()
    return msg.SerializeToString()
