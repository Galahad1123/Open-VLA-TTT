"""Explicit stateful Qwen2 execution, without hooks or mutation during recomputation."""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .attention import attention_mask


def language_step(
    language_model,
    connections,
    config,
    inputs,
    valid,
    visual,
    memory=None,
    write_mask=None,
    reset_mask=None,
):
    if memory is not None and len(memory) != len(connections):
        raise ValueError("Memory layer count mismatch")
    if valid.shape != inputs.shape[:2] or visual.shape != valid.shape:
        raise ValueError("VLM token masks must have shape (batch, tokens)")
    if not valid.any(1).all():
        raise ValueError("Each observation requires a valid VLM token")
    hidden = inputs.masked_fill(~valid[:, :, None], 0)
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
    rotary = language_model.rotary_emb(hidden, positions)
    mask = attention_mask(valid, dtype=hidden.dtype)
    updated = []
    use_checkpoint = config.enable_gradient_checkpointing and torch.is_grad_enabled()
    index = 0
    for i, layer in enumerate(language_model.layers):
        if config.memory_enabled and str(i) in connections:
            state = None if memory is None else memory[index]
            fn = partial(
                connections[str(i)],
                layer,
                valid=valid,
                visual=visual,
                rotary=rotary,
                state=state,
                write_mask=write_mask,
                reset_mask=reset_mask,
            )
            # Bind every layer/state by value: backward may recompute after this loop ends.
            hidden, new_state = (
                checkpoint(fn, hidden, use_reentrant=False) if use_checkpoint else fn(hidden)
            )
            updated.append(new_state)
            index += 1
        else:
            # Call forward directly to avoid a second, state-unaware HF checkpoint wrapper.
            fn = partial(
                layer.forward,
                attention_mask=mask,
                position_embeddings=rotary,
                position_ids=positions,
                use_cache=False,
            )
            hidden = checkpoint(fn, hidden, use_reentrant=False) if use_checkpoint else fn(hidden)
            hidden = hidden.masked_fill(~valid[:, :, None], 0)
    output = language_model.norm(hidden).masked_fill(~valid[:, :, None], 0)
    return output, tuple(updated) if config.memory_enabled else None
