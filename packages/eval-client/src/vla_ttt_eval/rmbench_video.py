"""Streaming, atomically published H.264 videos for RMBench."""

import subprocess
from pathlib import Path

import numpy as np


class H264Video:
    def __init__(self, path, shape, fps=10):
        self.path = Path(path)
        self.partial = self.path.with_suffix(".partial.mp4")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = (*shape, 3)
        self.frames = 0
        self.error = self.path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{shape[1]}x{shape[0]}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(self.partial),
            ],
            stdin=subprocess.PIPE,
            stderr=self.error,
        )

    def write(self, image):
        if image.dtype != np.uint8 or image.shape != self.shape:
            raise ValueError("Video frame shape/dtype changed")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.frames += 1

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.process.stdin.close()
            result = self.process.wait(timeout=60)
        except BaseException:
            self.process.kill()
            self.process.wait()
            raise
        finally:
            self.error.close()
        if kind is None:
            if result or not self.frames:
                raise RuntimeError(f"Video encoding failed: {self.partial}")
            self.partial.rename(self.path)
