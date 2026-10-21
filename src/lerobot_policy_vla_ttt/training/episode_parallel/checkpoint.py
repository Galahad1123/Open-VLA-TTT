"""Boundary checkpoints: shared slow state and independent rank RNG state."""

import json
import random
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_model

from .runtime import collective_error, distributed


def canonical(value):
    # Normalize JSON types without reordering mappings: augmentation names are
    # iterated in insertion order to map seeded draws to concrete transforms.
    return json.loads(json.dumps(value, default=str))


def normalize_saved_recipe(values):
    """Missing mode fields retain the historical full-episode behavior."""
    return {
        "training_mode": "alternating",
        "sampling_mode": "same_task",
        "update_unit": "episode",
        **values,
    }


def model_contract(policy, *, portable=False):
    values = asdict(policy.config)
    for name in ("device", "memory_training", "pretrained_path"):
        values.pop(name, None)
    if portable:
        # Portable CLI identity validates the complete VLM bytes separately.
        values.pop("vlm_model_name", None)
    return canonical(values)


def rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(path, policy, control, pre, post, accelerator, *, recipe, identity, totals):
    local_rng = rng_state(accelerator.device)
    states = [local_rng]
    if distributed():
        states = [None] * accelerator.num_processes
        dist.all_gather_object(states, local_rng)
    error = None
    if accelerator.is_main_process:
        try:
            path = Path(path)
            if path.exists():
                raise ValueError(f"Checkpoint already exists: {path}")
            temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
            temporary.mkdir(parents=True)
            model_dir = temporary / "pretrained_model"
            policy.save_pretrained(model_dir)
            for processor in (pre, post):
                if processor is not None:
                    processor.save_pretrained(model_dir)
            step = sum(control.counts.values())
            torch.save(
                {
                    "format": 1,
                    "world_size": accelerator.num_processes,
                    "recipe": asdict(recipe),
                    "identity": canonical(identity),
                    "model_contract": model_contract(policy, portable=identity.get("version") == 2),
                    "step": step,
                    "control": control.state_dict(),
                    "rank_rng": states,
                    "totals": totals,
                },
                temporary / "training.pt",
            )
            (temporary / "COMPLETE").write_text(f"{step}\n")
            temporary.rename(path)
        except Exception as exc:
            error = f"Checkpoint save failed: {exc}"
    collective_error(error, accelerator.device)
    accelerator.wait_for_everyone()


def load_checkpoint(path, policy, control, accelerator, *, recipe, identity):
    """Load only trusted local checkpoints; training.pt contains Python RNG state."""
    error, state = None, None
    try:
        path = Path(path)
        if not (path / "COMPLETE").is_file():
            raise ValueError("Incomplete checkpoint")
        state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        state["recipe"] = normalize_saved_recipe(state["recipe"])
        checks = {
            "format": 1,
            "world_size": accelerator.num_processes,
            "recipe": asdict(recipe),
            "identity": canonical(identity),
            "model_contract": model_contract(policy, portable=identity.get("version") == 2),
        }
        for key, expected in checks.items():
            if state[key] != expected:
                raise ValueError(f"Checkpoint {key} mismatch")
        if (
            int((path / "COMPLETE").read_text()) != state["step"]
            or state["step"] != sum(state["control"]["counts"].values())
            or len(state["rank_rng"]) != accelerator.num_processes
        ):
            raise ValueError("Inconsistent checkpoint counters")
    except Exception as exc:
        error = f"Checkpoint load failed: {exc}"
    collective_error(error, accelerator.device)
    error = None
    try:
        load_model(
            policy,
            str(path / "pretrained_model/model.safetensors"),
            strict=True,
            device="cpu" if accelerator.device.type == "cpu" else str(accelerator.device),
        )
        control.load_state_dict(state["control"])
        restore_rng(state["rank_rng"][accelerator.process_index], accelerator.device)
    except Exception as exc:
        error = f"Checkpoint restore failed: {exc}"
    collective_error(error, accelerator.device)
    return state["step"], state["totals"]
