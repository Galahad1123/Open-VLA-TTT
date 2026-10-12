"""Explicit synchronous data parallelism at outer-update boundaries.

Do not pass the model to Accelerator.prepare: autograd/DDP hooks would synchronize
inside unequal-length episodes. Gradients already include 1/global_batch_size.
"""

import torch
import torch.distributed as dist


def distributed():
    return dist.is_available() and dist.is_initialized()


def collective_error(error, device):
    """Propagate ordinary local failures once every live rank reaches a boundary.

    A crashed process/failed communicator still relies on the launcher's timeout.
    """
    if distributed():
        flag = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
            raise RuntimeError(f"Rank failure: {errors}")
    elif error is not None:
        raise RuntimeError(error)


@torch.no_grad()
def broadcast_model(policy):
    if distributed():
        for tensor in policy.state_dict().values():
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def synchronize_gradients(policy, *, bucket_bytes=25 * 1024 * 1024):
    """SUM slow gradients, retaining None for globally unused parameters.

    Presence and finite checks have identical collective order on all ranks.
    Buckets include slices of large parameters, bounding extra memory usage.
    """
    if bucket_bytes < 4:
        raise ValueError("bucket_bytes must be at least 4")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No active parameters")
    device = parameters[0].device
    invalid = torch.zeros((), dtype=torch.int64, device=device)
    for p in parameters:
        if p.grad is not None:
            if p.grad.is_sparse:
                invalid.fill_(1)
            else:
                invalid.add_((~p.grad.isfinite().all()).long())
    if distributed():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if invalid.item():
        raise FloatingPointError("Nonfinite or sparse slow gradient on at least one rank")
    if not distributed():
        return
    present = torch.tensor(
        [p.grad is not None for p in parameters], device=device, dtype=torch.int64
    )
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    # .tolist() synchronizes once rather than once per parameter on CUDA.
    present = present.tolist()
    gradients = []
    for p, used in zip(parameters, present, strict=True):
        if used:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad = p.grad.contiguous()
            gradients.append(p.grad)
    buckets, size, dtype = [], 0, None

    def flush():
        if not buckets:
            return
        buffer = torch.cat(buckets)
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        offset = 0
        for view in buckets:
            view.copy_(buffer[offset : offset + view.numel()])
            offset += view.numel()

    for grad in gradients:
        if dtype is not None and dtype != grad.dtype:
            flush()
            buckets, size = [], 0
        dtype = grad.dtype
        # A contiguous grad allocation is needed for writable flattened views.
        if not grad.is_contiguous():
            raise ValueError("Explicit reducer requires contiguous gradients")
        flat = grad.view(-1)
        capacity = max(1, bucket_bytes // grad.element_size())
        offset = 0
        while offset < flat.numel():
            n = min(capacity - size, flat.numel() - offset)
            buckets.append(flat[offset : offset + n])
            size += n
            offset += n
            if size == capacity:
                flush()
                buckets, size = [], 0
    flush()
