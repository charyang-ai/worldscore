"""PyTorch selective scan used by VFIMamba when the Mamba HIP kernel is absent.

This is the real-valued reference recurrence from mamba_ssm 2.2.4
(Apache-2.0): a sequential scan, not a different model. VFIMamba calls it
with grouped B/C of shape (batch, groups, state, length).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import repeat


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, length = u.shape
    dstate = A.shape[1]
    x = A.new_zeros((batch, dim, dstate))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))

    if B.dim() == 2:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B.float(), u)
    elif B.dim() == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B.float(), u)
    else:
        B = repeat(B.float(), "B G N L -> B (G H) N L", H=dim // B.shape[1])
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if C.dim() == 4:
        C = repeat(C.float(), "B G N L -> B (G H) N L", H=dim // C.shape[1])
    else:
        C = C.float()

    ys = []
    last_state = None
    for i in range(length):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 2:
            y = torch.einsum("bdn,dn->bd", x, C)
        elif C.dim() == 3:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        else:
            y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == length - 1:
            last_state = x
        ys.append(y.real if torch.is_complex(y) else y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.float().unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    if return_last_state:
        return out, last_state
    return out
