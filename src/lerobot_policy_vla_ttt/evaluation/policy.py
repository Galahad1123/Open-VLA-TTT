"""Evo-1 inference boundary. Normalization and action queue stay server-side."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.evo1.modeling_evo1 import Evo1Policy

from ..cli import make_pre_post_processors
from ..modeling_vla_ttt import VLATTTPolicy
from ..models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy


def checkpoint_identity(path):
    identities = {}
    for item in sorted(Path(path).iterdir()):
        if item.is_file() and item.suffix in {".json", ".safetensors"}:
            with item.open("rb") as stream:
                identities[item.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return identities


def validate_config(config, benchmark, n_action_steps):
    if benchmark not in {"libero", "robomme"}:
        raise ValueError("Unknown benchmark")
    action_dim = 7 if benchmark == "libero" else 8
    if config.output_features["action"].shape != (action_dim,):
        raise ValueError(f"Checkpoint action dimension does not match {benchmark}: {action_dim}")
    if config.input_features["observation.state"].shape != (8,):
        raise ValueError("Checkpoint state dimension must be 8")
    cameras = {
        key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL
    }
    if cameras != {"observation.images.image", "observation.images.image2"}:
        raise ValueError("Checkpoint camera names must be image and image2")
    if not 1 <= n_action_steps <= config.chunk_size:
        raise ValueError("n_action_steps must be in [1, chunk_size]")
    if config.n_obs_steps != 1 or config.rtc_config is not None:
        raise ValueError("This synchronous baseline requires one observation and no RTC")
    return action_dim


def observation_batch(observation):
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite state vector of shape (8,)")
    task = observation["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Expected nonempty task instruction")
    batch = {"observation.state": torch.from_numpy(state.copy()).unsqueeze(0), "task": [task]}
    for name in ("image", "image2"):
        image = observation["images"][name]
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape != (256, 256, 3)
        ):
            raise ValueError("Expected uint8 HWC image (256, 256, 3)")
        batch[f"observation.images.{name}"] = (
            torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        )
    return batch


class Evo1Inference:
    def __init__(self, checkpoint, implementation, benchmark, device="cuda", n_action_steps=10):
        path = Path(checkpoint).resolve()
        if not path.is_dir():
            raise ValueError("Use a downloaded, pinned local checkpoint directory")
        config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        action_dim = validate_config(config, benchmark, n_action_steps)
        config.device = device
        config.n_action_steps = n_action_steps
        if implementation == "native" and config.type == "evo1":
            self.policy = Evo1Policy.from_pretrained(path, config=config, local_files_only=True)
        elif implementation == "local" and config.type == "evo1":
            self.policy = VLATTTPolicy.from_evo1_pretrained(
                path, config=config, local_files_only=True
            )
        elif implementation == "local" and config.type == "vla_ttt":
            self.policy = VLATTTPolicy.from_pretrained(path, config=config, local_files_only=True)
        elif implementation == "evo1_groot" and config.type == "evo1_groot":
            self.policy = Evo1GrootPolicy.from_pretrained(
                path, config=config, local_files_only=True
            )
        else:
            raise ValueError(
                "Implementation must match checkpoint type: native=evo1, "
                "local=evo1/vla_ttt, evo1_groot=evo1_groot"
            )
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.metadata = {
            "benchmark": benchmark,
            "action_dim": action_dim,
            "state_dim": 8,
            "action_space": "eef_delta" if benchmark == "libero" else "joint_angle",
            "implementation": implementation,
            "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_identity(path),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("lerobot", "torch", "lerobot_policy_vla_ttt")
            },
            "device": device,
            "n_action_steps": n_action_steps,
            "chunk_size": config.chunk_size,
            "observation_mode": "latest_frame",
            "actions_normalized": False,
            "config": json.loads((path / "config.json").read_text()),
        }

    def reset(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @torch.inference_mode()
    def infer(self, observation):
        batch = self.pre(observation_batch(observation))
        action = self.post(self.policy.select_action(batch))
        return action[0].detach().cpu().numpy().astype(np.float32)
