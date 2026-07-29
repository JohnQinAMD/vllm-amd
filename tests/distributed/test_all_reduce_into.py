# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers import linear


def test_aiter_adapter_detects_caller_output_support():
    supported = object.__new__(AiterCustomAllreduce)
    supported._impl = SimpleNamespace(supports_custom_all_reduce_out=True)
    unsupported = object.__new__(AiterCustomAllreduce)
    unsupported._impl = SimpleNamespace()

    assert supported.supports_custom_all_reduce_out
    assert not unsupported.supports_custom_all_reduce_out


def test_aiter_adapter_forwards_caller_owned_output():
    calls = []
    output = torch.empty(8)

    def custom_all_reduce(input_, **kwargs):
        calls.append((input_, kwargs))
        return kwargs["out"]

    implementation = SimpleNamespace(custom_all_reduce=custom_all_reduce)
    communicator = SimpleNamespace(_impl=implementation)
    input_ = torch.empty_like(output)

    result = AiterCustomAllreduce.custom_all_reduce(
        communicator,
        input_,
        out=output,
    )

    assert result is output
    assert calls == [(input_, {"out": output})]


def test_aiter_adapter_does_not_pass_output_to_older_aiter():
    calls = []
    expected = torch.empty(8)

    def custom_all_reduce(input_):
        calls.append(input_)
        return expected

    implementation = SimpleNamespace(custom_all_reduce=custom_all_reduce)
    communicator = SimpleNamespace(_impl=implementation)
    input_ = torch.empty_like(expected)

    result = AiterCustomAllreduce.custom_all_reduce(communicator, input_)

    assert result is expected
    assert calls == [input_]


def test_cuda_communicator_uses_aiter_output_contract():
    calls = []
    output = torch.empty(8)

    def custom_all_reduce(input_, **kwargs):
        calls.append((input_, kwargs))
        return kwargs["out"]

    aiter = SimpleNamespace(
        disabled=False,
        supports_custom_all_reduce_out=True,
        should_custom_ar=lambda input_: True,
        custom_all_reduce=custom_all_reduce,
    )
    communicator = SimpleNamespace(
        aiter_ar_comm=aiter,
        all_reduce=lambda input_: (_ for _ in ()).throw(
            AssertionError("fallback should not run")
        ),
    )
    input_ = torch.empty_like(output)

    result = CudaCommunicator.all_reduce_into(communicator, input_, output)

    assert result is output
    assert calls == [(input_, {"out": output})]


def test_cuda_communicator_fallback_copies_result():
    input_ = torch.arange(8, dtype=torch.float32)
    output = torch.empty_like(input_)
    communicator = SimpleNamespace(
        aiter_ar_comm=None,
        all_reduce=lambda input_: input_ + 1,
    )

    result = CudaCommunicator.all_reduce_into(communicator, input_, output)

    assert result is output
    torch.testing.assert_close(output, input_ + 1)


def test_cuda_communicator_falls_back_for_older_aiter():
    input_ = torch.arange(8, dtype=torch.float32)
    output = torch.empty_like(input_)
    aiter = SimpleNamespace(
        disabled=False,
        supports_custom_all_reduce_out=False,
        should_custom_ar=lambda input_: True,
        custom_all_reduce=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("older AITER must not receive an output argument")
        ),
    )
    communicator = SimpleNamespace(
        aiter_ar_comm=aiter,
        all_reduce=lambda input_: input_ + 1,
    )

    result = CudaCommunicator.all_reduce_into(communicator, input_, output)

    assert result is output
    torch.testing.assert_close(output, input_ + 1)


def test_group_coordinator_world_size_one_preserves_output_identity():
    input_ = torch.arange(8, dtype=torch.float32)
    output = torch.empty_like(input_)
    coordinator = SimpleNamespace(world_size=1)

    result = GroupCoordinator.all_reduce_into(coordinator, input_, output)

    assert result is output
    torch.testing.assert_close(output, input_)


def test_group_coordinator_rejects_mismatched_output():
    coordinator = SimpleNamespace(world_size=1)

    with pytest.raises(ValueError, match="all-reduce output"):
        GroupCoordinator.all_reduce_into(
            coordinator,
            torch.empty(8),
            torch.empty(7),
        )


def test_row_parallel_linear_passes_stable_output(monkeypatch):
    calls = []
    input_ = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    partial = input_ + 1
    output = torch.empty_like(partial)
    layer = SimpleNamespace(
        input_is_parallel=True,
        tp_rank=0,
        skip_bias_add=False,
        bias=None,
        quant_method=SimpleNamespace(
            apply=lambda layer_, input_parallel, bias: partial
        ),
        reduce_results=True,
        tp_size=8,
        return_bias=False,
    )

    def all_reduce_into(input_parallel, destination):
        calls.append((input_parallel, destination))
        destination.copy_(input_parallel)
        return destination

    monkeypatch.setattr(
        linear,
        "tensor_model_parallel_all_reduce_into",
        all_reduce_into,
    )

    result = linear.RowParallelLinear.forward(layer, input_, output=output)

    assert result is output
    assert calls == [(partial, output)]
    torch.testing.assert_close(output, partial)


def test_row_parallel_linear_rejects_mismatched_output():
    input_ = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    layer = SimpleNamespace(
        input_is_parallel=True,
        tp_rank=0,
        skip_bias_add=False,
        bias=None,
        quant_method=SimpleNamespace(
            apply=lambda layer_, input_parallel, bias: input_parallel
        ),
        reduce_results=True,
        tp_size=8,
        return_bias=False,
    )

    mismatched = torch.empty((1, 7), dtype=torch.float32)

    with pytest.raises(ValueError, match="row-parallel output"):
        linear.RowParallelLinear.forward(layer, input_, output=mismatched)
