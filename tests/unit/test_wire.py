"""Our wire codec must be the exact inverse of the pinned SDK's own converters."""

import numpy as np
import pytest
import tinker.types as t
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.proto.response_conv import deserialize_forward_backward_output, deserialize_sample_response
import zstandard

from zroform.wire import (Sequence, decode_forward_backward, encode_forward_backward_output,
                          encode_sample_response)


def _sdk_request(loss_fn="cross_entropy", config=None):
    data = [
        t.Datum(model_input=t.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}),
        t.Datum(model_input=t.ModelInput.from_ints([7, 8]),
                loss_fn_inputs={"target_tokens": [8, 9], "weights": [1.0, 0.5]}),
    ]
    req = t.ForwardBackwardRequest(forward_backward_input=t.ForwardBackwardInput(
        data=data, loss_fn=loss_fn, loss_fn_config=config), model_id="S:train:0", seq_id=3)
    return forward_backward_request_to_proto(req).SerializeToString()


def test_decodes_what_the_sdk_sends():
    req = decode_forward_backward(_sdk_request(config={"beta": 0.1, "mode": "x"}), None)
    assert (req.model_id, req.seq_id, req.loss_fn, req.forward_only) == ("S:train:0", 3, "cross_entropy", False)
    assert req.config == {"beta": 0.1, "mode": "x"}
    assert req.data[0].tokens.tolist() == [1, 2, 3]
    assert req.data[1].inputs["target_tokens"].tolist() == [8, 9]
    np.testing.assert_allclose(req.data[1].inputs["weights"], [1.0, 0.5])


def test_decodes_zstd_compressed_body():
    body = zstandard.ZstdCompressor().compress(_sdk_request())
    assert decode_forward_backward(body, "zstd").data[0].tokens.tolist() == [1, 2, 3]


def test_sdk_reads_our_forward_backward_output():
    outs = [{"logprobs": np.array([-0.5, -1.0, -2.0], np.float32)}, {"logprobs": np.array([-3.0], np.float32)}]
    got = deserialize_forward_backward_output(encode_forward_backward_output(outs, {"loss:sum": 6.5}))
    assert got.metrics == {"loss:sum": 6.5}
    assert [o["logprobs"].tolist() for o in got.loss_fn_outputs] == [[-0.5, -1.0, -2.0], [-3.0]]


def test_rejects_metric_keys_without_reduction():
    with pytest.raises(ValueError):
        encode_forward_backward_output([], {"loss": 1.0})


def test_sdk_reads_our_sample_response():
    seqs = [Sequence([5, 6, 7], [-0.1, -0.2, -0.3], "stop"), Sequence([9], [-1.5], "length")]
    got = deserialize_sample_response(encode_sample_response(seqs, [None, -0.7, -0.8]))
    assert [s.tokens for s in got.sequences] == [[5, 6, 7], [9]]
    assert [s.stop_reason for s in got.sequences] == ["stop", "length"]
    np.testing.assert_allclose(got.sequences[0].logprobs, [-0.1, -0.2, -0.3], rtol=1e-6)
    assert got.prompt_logprobs[0] is None and got.prompt_logprobs[1] == pytest.approx(-0.7)
