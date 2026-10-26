"""Bounded-memory dual-camera H264 recording via system ffmpeg."""

import subprocess

import numpy as np


class Video:
    def __init__(self, path, fps):
        if path.exists() or path.with_name(path.stem + ".partial.mp4").exists():
            raise FileExistsError(path)
        self.path = path
        self.temp = path.with_name(path.stem + ".partial.mp4")
        self.frames = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "512x256",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.temp),
            ],
            stdin=subprocess.PIPE,
        )

    def capture(self, observation):
        frame = np.concatenate(
            [observation["images"]["image"], observation["images"]["image2"]], axis=1
        )
        if frame.shape != (256, 512, 3) or frame.dtype != np.uint8:
            raise ValueError("Expected two uint8 256x256 cameras")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def finish(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise
        if code != 0:
            raise RuntimeError(f"H264 encoding failed: {self.path}, code={code}")
        self.temp.rename(self.path)
