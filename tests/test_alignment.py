# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for zero-valued source coordinates in alignment."""

import itertools

import pytest
import torch

from metrics import alignment


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("trunc", [None, 1.0, 4.0])
@pytest.mark.parametrize("zero_target", [-3.0, 3.0])
def test_zero_source_preserves_constant_residual(dtype, trunc, zero_target):
    source = torch.tensor([0., 1., 2.], dtype=dtype)
    target = torch.tensor([zero_target, 2., 4.], dtype=dtype)
    weight = torch.tensor([2., 1., 1.], dtype=dtype)
    scale, loss, _ = alignment.align(source, target, weight, trunc)
    residual = weight * (scale * source - target).abs()
    if trunc is not None:
        residual = residual.clamp_max(trunc)
    torch.testing.assert_close(scale, torch.tensor(2., dtype=dtype))
    torch.testing.assert_close(loss, residual.sum())


@pytest.mark.parametrize("trunc", [None, 1.0, 4.0])
def test_zero_source_cost_with_broadcast_weights(trunc):
    source = torch.tensor([[0., 1., 2.], [0., -1., -2.]])
    target = torch.tensor([[3., 2., 4.], [-3., -2., -4.]])
    weight = torch.tensor([2., 1., 1.])
    scale, loss, _ = alignment.align(source, target, weight, trunc)
    residual = weight * (scale[:, None] * source - target).abs()
    if trunc is not None:
        residual = residual.clamp_max(trunc)
    torch.testing.assert_close(loss, residual.sum(-1))


def test_zero_source_target_gradient_is_retained():
    source = torch.tensor([0., 1., 2.])
    target = torch.tensor([3., 2., 4.], requires_grad=True)
    _, loss, _ = alignment.align(source, target, torch.tensor([2., 1., 1.]))
    (gradient,) = torch.autograd.grad(loss, target)
    torch.testing.assert_close(gradient[0], torch.tensor(2.))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_affine_alignment_with_repeated_depths_minimizes_actual_cost(dtype):
    source = torch.tensor([0., 1., 3., 1., 1., 1.], dtype=dtype)
    target = torch.tensor([4., -1., 7., 1., 1., 2.], dtype=dtype)
    weight = torch.tensor([3., 3., 3., 4., 4., 4.], dtype=dtype)
    scale, shift = alignment.align_depth_affine(source, target, weight)
    actual = (weight * (source * scale + shift - target).abs()).sum()
    # An optimum of weighted absolute affine regression lies at two anchors.
    costs = []
    for i, j in itertools.combinations(range(len(source)), 2):
        if source[i] != source[j]:
            a = (target[j] - target[i]) / (source[j] - source[i])
            b = target[i] - a * source[i]
            costs.append((weight * (source * a + b - target).abs()).sum())
    torch.testing.assert_close(actual, torch.stack(costs).min())
