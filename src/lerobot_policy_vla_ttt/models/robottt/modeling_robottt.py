"""Independent policy using established Evo-1 I/O and explicit trajectory state."""

from dataclasses import asdict

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

from ...checkpoint import checkpoint_policy_type
from ...compatibility import ensure_supported_lerobot_version
from ...evo1.modeling_evo1 import Evo1Policy
from ..evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from .configuration_robottt import RoboTTTConfig
from .model import RoboTTTModel


class RoboTTTPolicy(Evo1GrootPolicy):
    """Reuse the baseline's I/O adapter, not its memory-free forward lifecycle."""

    config_class = RoboTTTConfig
    name = "robottt"
    _runtime_overrides = Evo1GrootPolicy._runtime_overrides | {"memory_training"}

    def __init__(self, config, *, vlm_hub_kwargs=None, **kwargs):
        ensure_supported_lerobot_version()
        if not isinstance(config, RoboTTTConfig):
            raise TypeError("RoboTTTPolicy requires RoboTTTConfig")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        if len(config.image_features) > config.max_views:
            raise ValueError("Number of camera streams exceeds max_views")
        self.model = RoboTTTModel(config, vlm_hub_kwargs=vlm_hub_kwargs)
        if config.memory_training == "memory_only":
            self.model.requires_grad_(False)
            self.model.action_head.memories.requires_grad_(True)
            self.model.action_head.tokens.registers.requires_grad_(True)
        else:
            self.model.set_finetune_flags()
        self._keep_frozen_embedder_eval()
        self.init_rtc_processor()
        self.reset()

    @property
    def _tracks_vlm_gradients(self):
        return self.config.memory_training == "joint" and super()._tracks_vlm_gradients

    def reset(self):
        super().reset()
        self.model.action_head.reset()

    def forward_step(
        self, batch, memory=None, *, write_mask=None, supervision_mask=None, reset_mask=None
    ):
        """Return an unnormalized loss, valid scalar count, and explicit next state.

        Context-only rows use supervision_mask=False. Their action input is zero
        and action_mask=False, so labels cannot enter memory accidentally.
        """
        if not self.config.memory_enabled and self.config.memory_training == "memory_only":
            raise ValueError(
                "memory_only training requires memory_enabled=True; bypass is evaluation-only"
            )
        prompts = self._normalize_task_batch(batch)
        images, image_masks = self._collect_image_batches(batch)
        states, _ = self._prepare_state(batch)
        if "action" not in batch:
            if supervision_mask is None or torch.as_tensor(supervision_mask).any():
                raise ValueError("Missing action is only valid for context-only rows")
            batch = dict(
                batch,
                action=torch.zeros(
                    len(states), self.config.chunk_size, self._env_action_dim, device=states.device
                ),
            )
        actions, mask = self._prepare_actions(batch)
        if supervision_mask is not None:
            supervision_mask = supervision_mask.to(device=mask.device, dtype=torch.bool)
            if supervision_mask.shape != (len(states),):
                raise ValueError("supervision_mask must have shape (batch,)")
            mask = mask & supervision_mask[:, None, None]
        ids = self._get_embodiment_ids(batch, len(states))
        with self._maybe_autocast():
            context, context_mask = self._compute_fused_tokens(prompts, images, image_masks)
            pred, noise, memory = self.model.action_head.training_step(
                context,
                states,
                actions,
                action_mask=mask,
                embodiment_id=ids,
                context_mask=context_mask,
                memory=memory,
                write_mask=write_mask,
                reset_mask=reset_mask,
            )
        target = (actions.float() - noise.float()).masked_fill(~mask, 0)
        error = (pred.float() - target).square().masked_fill(~mask, 0)
        return error.sum(), mask.sum(), memory

    def forward(self, batch, reduction="mean"):
        if reduction != "mean":
            raise ValueError("RoboTTT supports mean loss; use forward_step for sequential training")
        numerator, count, _ = self.forward_step(batch)
        loss = numerator / count.clamp_min(1)
        return loss, {"loss": loss.detach().item(), "supervised_scalars": count.item()}

    def initialize_from_baseline(self, baseline):
        if baseline.config.type != "evo1_groot":
            raise ValueError("Initialization requires an evo1_groot baseline")
        allowed = self._runtime_overrides | {"num_register_tokens"}
        actual = asdict(self.config)
        changed = [
            k for k, v in asdict(baseline.config).items() if k not in allowed and actual[k] != v
        ]
        if list(self.config.image_features) != list(baseline.config.image_features):
            changed.append("camera order")
        if changed:
            raise ValueError(f"Baseline configuration mismatch: {changed}")
        report = self.model.action_head.initialize_from_baseline(baseline.model.action_head)
        self.model.embedder.load_state_dict(baseline.model.embedder.state_dict(), strict=True)
        self.reset()
        return report

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if kwargs.get("strict", True) is not True:
            raise ValueError("RoboTTT checkpoint loading requires strict=True")
        kwargs["strict"] = True
        hub = {
            k: kwargs[k]
            for k in ("revision", "local_files_only", "cache_dir", "token")
            if k in kwargs
        }
        kind = checkpoint_policy_type(pretrained_name_or_path, **hub)
        if kind != cls.name:
            raise ValueError(f"Checkpoint type {kind!r}; expected 'robottt'")
        override = kwargs.get("config")
        if override is not None:
            if not isinstance(override, RoboTTTConfig):
                raise TypeError("RoboTTTPolicy requires RoboTTTConfig")
            saved = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **hub)
            values = asdict(override)
            changed = [
                k
                for k, v in asdict(saved).items()
                if k not in cls._runtime_overrides and values[k] != v
            ]
            if list(saved.image_features) != list(override.image_features):
                changed.append("camera order")
            if changed:
                raise ValueError(f"Checkpoint configuration mismatch: {changed}")
        # Skip the baseline-family loader; retain Evo1's Hub/VLM forwarding.
        return Evo1Policy.from_pretrained.__func__(cls, pretrained_name_or_path, **kwargs)
