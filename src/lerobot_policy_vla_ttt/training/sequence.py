"""Episode-bounded sequential windows; shuffling windows never shuffles time."""

import torch
from torch.utils.data import Dataset, default_collate


class TrajectoryWindows(Dataset):
    def __init__(self, frames, episode_ranges, *, length, stride):
        if length < 1 or stride < 1:
            raise ValueError("length and stride must be positive")
        self.frames, self.length, self.stride = frames, length, stride
        self.windows = []
        previous_end = 0
        for start, end in episode_ranges:
            if not 0 <= previous_end <= start < end <= len(frames):
                raise ValueError("Episode ranges must be ordered, disjoint and within dataset")
            self.windows.extend((i, end) for i in range(start, end, length * stride))
            previous_end = end
        if not self.windows:
            raise ValueError("No trajectory windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        start, end = self.windows[index]
        indices = list(range(start, min(end, start + self.length * self.stride), self.stride))
        valid = torch.arange(self.length) < len(indices)
        indices.extend([indices[-1]] * (self.length - len(indices)))
        return {"frames": [self.frames[i] for i in indices], "valid": valid}


def collate_windows(windows):
    return {
        "frames": [
            default_collate([window["frames"][t] for window in windows])
            for t in range(len(windows[0]["frames"]))
        ],
        "valid_mask": torch.stack([window["valid"] for window in windows]),
    }
