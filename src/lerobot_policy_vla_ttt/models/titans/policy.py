"""Shared explicit VLM state lifecycle and LeRobot I/O for MAC/MAG/MAL.

Only the existing stateless batch adapters are reused. This policy does not
inherit the baseline's forward, model construction, or inference lifecycle.
"""

from collections import deque
from dataclasses import asdict

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

from ...checkpoint import checkpoint_policy_type
from ...compatibility import ensure_supported_lerobot_version
from ...evo1.modeling_evo1 import Evo1Policy
from ..evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from .configuration import TitansConfig
from .model import TitansModel


class TitansPolicy(PreTrainedPolicy):
    config_class = TitansConfig
    name = "titans_internal"
    requires_fast_weight_grad = True
    _runtime_overrides = Evo1GrootPolicy._runtime_overrides | {"memory_training"}
    _camera_keys = Evo1Policy._camera_keys
    _env_action_dim = Evo1Policy._env_action_dim
    _compute_dtype = Evo1Policy._compute_dtype
    _device = Evo1Policy._device
    _normalize_task_batch = Evo1Policy._normalize_task_batch
    _prepare_state = Evo1Policy._prepare_state
    _prepare_actions = Evo1Policy._prepare_actions
    _prepare_inference_action_mask = Evo1Policy._prepare_inference_action_mask
    _get_embodiment_ids = Evo1Policy._get_embodiment_ids
    _collect_image_batches = Evo1Policy._collect_image_batches
    get_optim_params = Evo1Policy.get_optim_params
    _amp_enabled = Evo1GrootPolicy._amp_enabled
    _maybe_autocast = Evo1GrootPolicy._maybe_autocast

    def __init__(self, config, *, vlm_hub_kwargs=None, **kwargs):
        ensure_supported_lerobot_version()
        if type(config) is not self.config_class:
            raise TypeError(f"{self.name} requires {self.config_class.__name__}")
        super().__init__(config)
        config.validate_features()
        if len(config.image_features) > config.max_views:
            raise ValueError("Number of camera streams exceeds max_views")
        self.model = TitansModel(config, self.connection_class, vlm_hub_kwargs=vlm_hub_kwargs)
        self.reset()
        self.train()

    def supports_rtc(self):
        return False

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.inference_state = None

    def _encode(self, batch, memory, write_mask=None, reset_mask=None):
        images, image_masks = self._collect_image_batches(batch)
        prompts = self._normalize_task_batch(batch)
        # Frozen weights must still propagate the outer action gradient to memory.
        context, mask, updated = self.model.embedder.encode(
            images, image_masks, prompts, memory, write_mask, reset_mask
        )
        return context.to(device=self._device, dtype=self._compute_dtype), mask, updated

    def forward_step(
        self,
        batch,
        memory=None,
        *,
        write_mask=None,
        supervision_mask=None,
        reset_mask=None,
    ):
        if not self.config.memory_enabled and self.config.memory_training == "memory_only":
            raise ValueError("memory_only training requires memory_enabled=True")
        states, _ = self._prepare_state(batch)
        if "action" not in batch:
            if supervision_mask is None or torch.as_tensor(supervision_mask).any():
                raise ValueError("Missing action is only valid for context-only rows")
            batch = dict(
                batch,
                action=torch.zeros(
                    len(states), self.config.chunk_size, self._env_action_dim, device=self._device
                ),
            )
        actions, action_mask = self._prepare_actions(batch)
        if supervision_mask is not None:
            if supervision_mask.shape != (len(states),):
                raise ValueError("supervision_mask must have shape (batch,)")
            action_mask &= supervision_mask.to(device=self._device, dtype=torch.bool)[:, None, None]
        ids = self._get_embodiment_ids(batch, len(states))
        with self._maybe_autocast():
            context, mask, updated = self._encode(batch, memory, write_mask, reset_mask)
            pred, noise = self.model.action_head(
                context,
                states,
                actions,
                action_mask=action_mask,
                embodiment_id=ids,
                context_mask=mask,
            )
        pred = pred.view_as(actions).float()
        target = actions.float() - noise.float()
        error = (pred - target).square().masked_fill(~action_mask, 0)
        return error.sum(), action_mask.sum(), updated

    def forward(self, batch, reduction="mean"):
        if reduction != "mean":
            raise ValueError("Use forward_step for sequential, unnormalized losses")
        numerator, count, _ = self.forward_step(batch)
        loss = numerator / count.clamp_min(1)
        return loss, {"loss": loss.detach().item(), "supervised_scalars": count.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError("Titans policies do not support RTC sampling arguments")
        self.eval()
        states, _ = self._prepare_state(batch)
        ids = self._get_embodiment_ids(batch, len(states))
        action_mask = self._prepare_inference_action_mask(len(states))
        with self._maybe_autocast():
            context, mask, candidate = self._encode(batch, self.inference_state)
            actions = self.model.action_head.get_action(
                context,
                states,
                action_mask=action_mask,
                embodiment_id=ids,
                context_mask=mask,
            )
        # Atomic commit: a failed VLM/action pass cannot advance episode memory.
        self.inference_state = (
            None if candidate is None else tuple(state.detach() for state in candidate)
        )
        return actions.view(len(states), self.config.chunk_size, self.config.max_action_dim).float()

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        self.eval()
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def initialize_from_baseline(self, baseline):
        if baseline.config.type != "evo1_groot":
            raise ValueError("Initialization requires an evo1_groot checkpoint")
        actual = asdict(self.config)
        changed = [
            k
            for k, v in asdict(baseline.config).items()
            if k not in self._runtime_overrides and actual[k] != v
        ]
        if list(self.config.image_features) != list(baseline.config.image_features):
            changed.append("camera order")
        if changed:
            raise ValueError(f"Baseline configuration mismatch: {changed}")
        source, target = baseline.model.state_dict(), self.model.state_dict()
        new = [k for k in target if k.startswith("embedder.connections.")]
        copied = [k for k in target if k not in new]
        if set(copied) != set(source):
            raise ValueError("Baseline parameter names do not match")
        if any(target[k].shape != source[k].shape for k in copied):
            raise ValueError("Baseline parameter shapes do not match")
        with torch.no_grad():
            for k in copied:
                target[k].copy_(source[k])
        self.reset()
        return {"copied": copied, "new": new}

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if kwargs.get("strict", True) is not True:
            raise ValueError("Memory checkpoints require strict=True")
        hub = {
            k: kwargs[k]
            for k in ("revision", "local_files_only", "cache_dir", "token")
            if k in kwargs
        }
        if checkpoint_policy_type(pretrained_name_or_path, **hub) != cls.name:
            raise ValueError(f"Checkpoint type must be {cls.name}")
        saved = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **hub)
        config = kwargs.pop("config", None)
        if config is None:
            config = saved
        if type(config) is not cls.config_class:
            raise TypeError("Checkpoint configuration class mismatch")
        values = asdict(config)
        changed = [
            k
            for k, v in asdict(saved).items()
            if k not in cls._runtime_overrides and values[k] != v
        ]
        if list(saved.image_features) != list(config.image_features):
            changed.append("camera order")
        if changed:
            raise ValueError(f"Checkpoint configuration mismatch: {changed}")
        kwargs["strict"] = True
        kwargs.setdefault("vlm_hub_kwargs", {k: v for k, v in hub.items() if k != "revision"})
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)
