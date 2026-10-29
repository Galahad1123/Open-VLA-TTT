"""Display-only sequence progress; never touches RNG, tensors or training state."""

import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean


@dataclass
class Output:
    terminal: object
    log: object
    tty: bool
    columns: object = lambda: shutil.get_terminal_size(fallback=(80, 24)).columns


OUTPUT = ContextVar("sequence_progress_output", default=None)


def duration(seconds):
    seconds = max(0, round(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Progress:
    def __init__(
        self, recipe, start, end, *, mode="auto", enabled=True, output=None, clock=time.monotonic
    ):
        if mode not in {"auto", "bar", "plain", "off"}:
            raise ValueError("Unknown progress mode")
        self.output = (
            output or OUTPUT.get() or Output(self._stderr, lambda text: None, sys.stderr.isatty())
        )
        self.mode = ("bar" if self.output.tty else "plain") if mode == "auto" else mode
        self.enabled = enabled and mode != "off"
        self.recipe, self.step, self.end = recipe, start, end
        self.clock, self.started = clock, clock()
        self.last_summary, self.last_phase = None, None
        self.phase, self.loss = "-", None
        self.times = defaultdict(lambda: deque(maxlen=50))
        self.all_times, self.save_times = deque(maxlen=50), deque(maxlen=50)
        self.remaining = Counter(recipe.phase(s) for s in range(start, end))
        self.save_started = None
        self.last_saved = start
        self.width = 0

    @staticmethod
    def _stderr(text):
        sys.stderr.write(text)
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.emit("failed" if kind is not KeyboardInterrupt else "interrupted", force=True)
        else:
            self.finish()

    def eta(self):
        if len(self.all_times) < 3:
            return None, True
        fallback = mean(self.all_times)
        provisional = any(n and not self.times[p] for p, n in self.remaining.items())
        seconds = sum(
            n * (mean(self.times[p]) if self.times[p] else fallback)
            for p, n in self.remaining.items()
        )
        saves = self.end // self.recipe.save_every - self.step // self.recipe.save_every
        if self.end % self.recipe.save_every and self.end > self.step:
            saves += 1
        save_mean = mean(self.save_times) if self.save_times else 0
        seconds += saves * save_mean
        if self.step > self.last_saved and (
            self.step % self.recipe.save_every == 0 or self.step == self.end
        ):
            elapsed = 0 if self.save_started is None else self.clock() - self.save_started
            seconds += max(0, save_mean - elapsed)
        return seconds, provisional

    def update(self, metrics):
        self.step, self.phase, self.loss = metrics["step"], metrics["phase"], metrics["loss"]
        seconds = metrics["update_seconds"]
        self.times[self.phase].append(seconds)
        self.all_times.append(seconds)
        self.remaining[self.phase] -= 1
        self.emit(force=self.last_phase != self.phase)
        self.last_phase = self.phase

    def text(self, event):
        eta, provisional = self.eta()
        finish = "estimating"
        if eta is not None:
            stamp = datetime.now().astimezone() + timedelta(seconds=eta)
            finish = f"{duration(eta)} finish={stamp:%Y-%m-%d %H:%M:%S %Z%z}"
            if provisional:
                finish += " (provisional)"
        percent = 100 * self.step / self.end if self.end else 100
        speed = f"{mean(self.times[self.phase]):.2f}s/step" if self.times[self.phase] else "-"
        loss = f"{self.loss:.5g}" if self.loss is not None else "-"
        return (
            f"{event} {self.step}/{self.end} ({percent:.1f}%, recipe={self.recipe.steps}) "
            f"phase={self.phase} loss={loss} {speed} "
            f"elapsed={duration(self.clock() - self.started)} ETA={finish}"
        )

    def emit(self, event="training", force=False):
        if not self.enabled:
            return
        now = self.clock()
        summary = force or self.last_summary is None or now - self.last_summary >= 60
        text = self.text(event)
        if summary:
            try:
                self.output.log(text + "\n")
            except (OSError, ValueError):
                pass  # Metrics/checkpoints remain authoritative; diagnostics are best effort.
            self.last_summary = now
        try:
            if self.mode == "bar":
                limit = max(1, self.output.columns() - 1)
                if len(text) > limit:
                    eta, provisional = self.eta()
                    remaining = duration(eta) if eta is not None else "?"
                    finish = (
                        (datetime.now().astimezone() + timedelta(seconds=eta)).strftime("%H:%M")
                        if eta is not None
                        else "?"
                    )
                    speed = mean(self.all_times) if self.all_times else 0
                    loss = f"{self.loss:.3g}" if self.loss is not None else "-"
                    text = (
                        f"{self.step}/{self.end} {100 * self.step / max(1, self.end):.0f}% "
                        f"{self.phase} L={loss} {speed:.1f}s "
                        f"ETA={remaining}{'~' if provisional else ''} end={finish} "
                        f"elapsed={duration(self.clock() - self.started)}"
                    )
                    if event != "training":
                        text = event + " " + text
                text = text[:limit]
                self.output.terminal("\r" + text + " " * max(0, min(self.width, limit) - len(text)))
                self.width = len(text)
                if event in {"complete", "failed", "interrupted"}:
                    self.output.terminal("\n")
            elif summary:
                self.output.terminal(text + "\n")
        except (OSError, ValueError):
            pass  # A disconnected terminal must not stop training.

    def saving(self):
        self.save_started = self.clock()
        self.emit("saving", force=True)

    def saved(self):
        self.save_times.append(self.clock() - self.save_started)
        self.last_saved = self.step
        self.save_started = None
        self.emit("saved", force=True)

    def finish(self):
        self.emit("complete", force=True)
