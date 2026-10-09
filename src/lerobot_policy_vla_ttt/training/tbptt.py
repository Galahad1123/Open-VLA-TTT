"""Backward and release each segment; carry detached fast-weight values only."""

import torch


def train_sequence(
    policy,
    frames,
    optimizer,
    *,
    segment_length,
    valid_mask=None,
    write_mask=None,
    supervision_mask=None,
    reset_mask=None,
    grad_clip_norm=None,
    preprocess=None,
):
    if not frames or segment_length < 1:
        raise ValueError("Need nonempty frames and positive segment_length")
    batch = len(frames[0]["observation.state"])
    device = policy._device
    shape = (batch, len(frames))

    def mask(value, default):
        result = (
            torch.full(shape, default, dtype=torch.bool, device=device)
            if value is None
            else value.to(device=device, dtype=torch.bool)
        )
        if result.shape != shape:
            raise ValueError(f"Sequence masks must have shape {shape}")
        return result

    valid = mask(valid_mask, True)
    writes = mask(write_mask, True) & valid
    supervised = mask(supervision_mask, True) & valid
    resets = mask(reset_mask, False) & valid
    # Normalize all segment gradients by exactly the same valid scalar count.
    total_count = 0
    for t, frame in enumerate(frames):
        if "action" not in frame:
            if supervised[:, t].any():
                raise ValueError("Supervised frames require actions")
            continue
        _, action_mask = policy._prepare_actions(frame)
        total_count += int((action_mask & supervised[:, t, None, None]).sum().item())
    if total_count == 0:
        raise ValueError("A training window needs at least one supervised action")
    optimizer.zero_grad(set_to_none=True)
    memory, total_loss, segments = None, 0.0, 0
    for start in range(0, len(frames), segment_length):
        end = min(start + segment_length, len(frames))
        # The caller retains CPU inputs only. Device copies live for one segment.
        prepared = [preprocess(frame) if preprocess else frame for frame in frames[start:end]]
        segment_loss = None
        for t in range(start, end):
            numerator, _, memory = policy.forward_step(
                prepared[t - start],
                memory,
                write_mask=writes[:, t],
                supervision_mask=supervised[:, t],
                reset_mask=resets[:, t],
            )
            segment_loss = numerator if segment_loss is None else segment_loss + numerator
        if segment_loss.requires_grad:
            (segment_loss / total_count).backward()
        total_loss += segment_loss.detach().item()
        memory = None if memory is None else tuple(state.detach() for state in memory)
        del segment_loss, numerator, prepared
        segments += 1
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": total_loss / total_count,
        "supervised_scalars": total_count,
        "segments": segments,
    }, memory
