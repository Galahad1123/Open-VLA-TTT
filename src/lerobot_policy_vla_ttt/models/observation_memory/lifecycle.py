"""Shared physical observation clock; model-specific callbacks return candidate states."""

from dataclasses import replace

import torch

from .clock import ObservationClock


class HalfExecutionLifecycle:
    def _reset_observation_clock(self):
        self.inference_state = None
        self.clock = ObservationClock(self.config.n_action_steps)
        self._chunk = None
        self._chunk_start = None
        self._last_action = None
        self._last_action_step = -1

    def _validate_batch(self, batch):
        size = len(batch["observation.state"])
        if self._chunk is not None and size != len(self._chunk):
            raise ValueError("Memory batch changed; reset first")

    @torch.no_grad()
    def observe(self, batch, *, env_step):
        """Optional explicit midpoint delivery for clients executing chunks locally."""
        HalfExecutionLifecycle._validate_batch(self, batch)
        self.eval()
        if not self.clock.check(env_step):
            return
        if env_step % self.config.n_action_steps == 0:
            raise ValueError("Decision observations require predict_action_chunk before writing")
        if env_step % self.config.write_stride == 0:
            with self._maybe_autocast():
                candidate = self._write_candidate(batch)
            self.inference_state = (
                None if candidate is None else tuple(s.detach() for s in candidate)
            )
        self.clock.commit(env_step)

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, env_step=None, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError("RTC is unsupported")
        HalfExecutionLifecycle._validate_batch(self, batch)
        self.eval()
        step = (
            (0 if self._chunk_start is None else self._chunk_start + self.config.n_action_steps)
            if env_step is None
            else env_step
        )
        fresh = self.clock.check(step)
        if step % self.config.n_action_steps:
            raise ValueError("Replanning requires a decision boundary")
        if not fresh:
            if self._chunk_start != step:
                raise ValueError("No cached chunk for observation")
            return self._chunk.clone()
        with self._maybe_autocast():
            actions, candidate = self._sample_candidate(batch)
        self.inference_state = (
            None
            if candidate is None
            else tuple(replace(s.detach(), decision_index=s.decision_index + 1) for s in candidate)
        )
        self._chunk, self._chunk_start = actions.detach(), step
        self.clock.commit(step)
        return self._chunk.clone()

    @torch.no_grad()
    def select_action(self, batch, *, env_step=None, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError("RTC is unsupported")
        HalfExecutionLifecycle._validate_batch(self, batch)
        step = self._last_action_step + 1 if env_step is None else env_step
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Invalid environment step")
        if step == self._last_action_step:
            return self._last_action.clone()
        if step != self._last_action_step + 1:
            raise ValueError("select_action cannot skip environment steps")
        if step % self.config.n_action_steps == 0:
            self.predict_action_chunk(batch, env_step=step)
        else:
            self.observe(batch, env_step=step)
        action = self._chunk[:, step - self._chunk_start].clone()
        self._last_action, self._last_action_step = action, step
        return action.clone()
