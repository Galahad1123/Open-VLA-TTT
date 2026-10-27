"""Explicit observation writes and read-only action sampling for independent policies."""

from contextlib import nullcontext
from dataclasses import asdict, replace

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from ...compatibility import ensure_supported_lerobot_version
from ..titans.policy import TitansPolicy
from .lifecycle import HalfExecutionLifecycle
from .model import ObservationModel


class ObservationPolicy(HalfExecutionLifecycle, TitansPolicy):
    # Only stateless I/O adapters and strict checkpoint loading are inherited.
    def __init__(self, config, *, vlm_hub_kwargs=None, **kwargs):
        ensure_supported_lerobot_version()
        if type(config) is not self.config_class:
            raise TypeError("Wrong policy configuration")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        if len(config.image_features) > config.max_views:
            raise ValueError("Too many cameras")
        self.model = ObservationModel(config, self.head_class, vlm_hub_kwargs=vlm_hub_kwargs)
        self.reset()

    def reset(self):
        self._reset_observation_clock()

    def _observation(self, batch, memory=None, reset_mask=None):
        images, masks = self._collect_image_batches(batch)
        prompts = self._normalize_task_batch(batch)
        grad = any(p.requires_grad for p in self.model.embedder.parameters())
        with nullcontext() if grad else torch.no_grad():
            context, valid, layers = self.model.embedder.encode(images, masks, prompts)
        context = context.to(device=self._device, dtype=self._compute_dtype)
        valid = valid.to(self._device)
        features = tuple(x.to(context) for x in layers)
        cache = self.model.action_head.prepare_observation(features, valid, memory, reset_mask)
        return context, valid, cache

    def forward_step(
        self,
        batch,
        memory=None,
        *,
        write_mask=None,
        supervision_mask=None,
        reset_mask=None,
        loss_reduction="sum",
    ):
        if not self.config.memory_enabled and self.config.memory_training == "memory_only":
            raise ValueError("memory_only training requires enabled memory")
        if loss_reduction not in {"sum", "none"}:
            raise ValueError("Invalid loss reduction")
        states, _ = self._prepare_state(batch)
        supervised = (
            torch.ones(len(states), device=self._device, dtype=torch.bool)
            if supervision_mask is None
            else supervision_mask.to(self._device, dtype=torch.bool)
        )
        if supervised.shape != (len(states),):
            raise ValueError("Invalid supervision mask")
        with self._maybe_autocast():
            context, valid, cache = self._observation(batch, memory, reset_mask)
            if supervised.any():
                actions, mask = self._prepare_actions(batch)
                mask = mask & supervised[:, None, None]
                ids = self._get_embodiment_ids(batch, len(states))
                pred, noise = self.model.action_head.training_prediction(
                    context,
                    states,
                    actions,
                    cache,
                    action_mask=mask,
                    context_mask=valid,
                    embodiment_id=ids,
                )
                target = (actions.float() - noise.float()).masked_fill(~mask, 0)
                numerator = (pred.float() - target).square().masked_fill(~mask, 0).sum((1, 2))
                count = mask.sum((1, 2))
                if loss_reduction == "sum":
                    numerator, count = numerator.sum(), count.sum()
            else:
                # Observation-only events neither need labels nor sample action noise.
                shape = () if loss_reduction == "sum" else (len(states),)
                numerator = context.new_zeros(shape)
                count = torch.zeros(shape, device=context.device, dtype=torch.long)
            updated = self.model.action_head.commit(cache, write_mask)
        updated = tuple(
            replace(s, decision_index=s.decision_index + supervised.long()) for s in updated
        )
        return numerator, count, updated

    def _write_candidate(self, batch):
        _, _, cache = self._observation(batch, self.inference_state)
        return self.model.action_head.commit(cache)

    def _sample_candidate(self, batch):
        states, _ = self._prepare_state(batch)
        context, valid, cache = self._observation(batch, self.inference_state)
        actions = self.model.action_head.sample(
            context,
            states,
            cache,
            context_mask=valid,
            embodiment_id=self._get_embodiment_ids(batch, len(states)),
            action_mask=self._prepare_inference_action_mask(len(states)),
        )
        return actions, self.model.action_head.commit(cache)

    def initialize_from_baseline(self, baseline):
        if baseline.config.type != "evo1_groot":
            raise ValueError("Expected evo1_groot baseline")
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
        report = self.model.action_head.initialize_from_baseline(baseline.model.action_head)
        self.model.embedder.load_state_dict(baseline.model.embedder.state_dict(), strict=True)
        self.reset()
        return report
