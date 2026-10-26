"""scatter_sum and scatter_mean on the ROCm ATen kernels.

DROID-SLAM calls these two functions. The image has no rocThrust headers, so
the torch-scatter HIP extension does not build. torch.scatter_add_ is the
same reduction, already compiled for gfx1100 inside libtorch.
"""

from __future__ import annotations

import torch


def _broadcast(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim += src.dim()
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    for _ in range(index.dim(), src.dim()):
        index = index.unsqueeze(-1)
    return index.expand_as(src)


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    if dim < 0:
        dim += src.dim()
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = src.new_zeros(size)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = scatter_sum(src, index, dim, out, dim_size)
    ones = torch.ones_like(src)
    count = scatter_sum(ones, index, dim, None, out.size(dim))
    count = count.clamp(min=1)
    return out / count
