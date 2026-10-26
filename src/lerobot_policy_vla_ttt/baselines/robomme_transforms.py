"""CPU adaptation of official openpi image augmentation (augmax 0.4.1).

Pinned source: augmax-0.4.1.tar.gz SHA256
 d8e645203f535e243a3b16fb3634b10d4f168d1b9cfde4cda3892ab22bf31e8d.
Unlike torchvision ColorJitter, augmax defaults to p=.5 and hue=.1, adjusts
brightness/contrast in HSV value, and accidentally discards its saturation
adjustment. We preserve those observed semantics. Geometry uses torchvision
bilinear resampling rather than augmax's fused JAX coordinates; random streams
and final uint8 rounding differ. Geometry stays at dataset resolution 256
rather than openpi model resolution 224; bilinear zero border matches augmax.
This is an official-style CPU adaptation,
not bitwise equivalence. No horizontal/vertical flips are applied.
"""

import math

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms._functional_tensor import _hsv2rgb, _rgb2hsv


def _uniform(low, high):
    return float(torch.empty(()).uniform_(low, high))


class OfficialRoboMMEAugmentation:
    """Use torch's worker-seeded RNG; callers may control it with manual_seed."""

    def __call__(self, image, *, wrist=False):
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected RGB uint8 CHW image")
        image = image.to(torch.float32) / 255
        if not wrist:
            height, width = image.shape[-2:]
            crop_h, crop_w = int(height * 0.95), int(width * 0.95)
            top = int(torch.randint(height - crop_h + 1, ()))
            left = int(torch.randint(width - crop_w + 1, ()))
            image = F.resized_crop(
                image,
                top,
                left,
                crop_h,
                crop_w,
                [256, 256],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image = F.rotate(image, _uniform(-5, 5), interpolation=InterpolationMode.BILINEAR)
        if torch.rand(()) < 0.5:
            hsv = _rgb2hsv(image)
            hue, saturation, value = hsv.unbind(0)
            brightness = _uniform(-0.3, 0.3)
            value = (
                value * (1 + brightness)
                if brightness < 0
                else value * (1 - brightness) + brightness
            )
            slant = math.tan((_uniform(-0.4, 0.4) + 1) * math.pi / 4)
            p1 = slant / (2 * (1 + slant))
            value = torch.where(
                value < p1,
                value / slant,
                torch.where(
                    value > 1 - p1, value / slant + 1 - 1 / slant, slant * (value - 0.5) + 0.5
                ),
            )
            hue = (hue + _uniform(-0.1, 0.1)) % 1
            _uniform(-0.5, 0.5)  # Pinned augmax discards saturation adjustment result.
            image = _hsv2rgb(torch.stack((hue, saturation, value)))
        return image.clamp(0, 1).mul(255).round().to(torch.uint8)
