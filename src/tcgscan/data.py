from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def discover_samples(root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Treat every directory containing images as one expansion class."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    samples: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append((path, path.parent.name))
    if not samples:
        raise ValueError(f"No supported images found below: {root}")

    samples.sort(key=lambda item: str(item[0]))
    return samples, sorted({label for _, label in samples})


def split_samples(
    samples: list[tuple[Path, str]], seed: int = 42
) -> dict[str, list[tuple[Path, str]]]:
    """Make a repeatable per-class 80/10/10 split, where class size permits."""
    grouped: dict[str, list[tuple[Path, str]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    splits: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
        size = len(group)
        test_size = max(1, round(size * 0.1)) if size >= 3 else 0
        val_size = max(1, round(size * 0.1)) if size >= 2 else 0
        train_end = size - val_size - test_size
        splits["train"].extend(group[:train_end])
        splits["val"].extend(group[train_end : train_end + val_size])
        splits["test"].extend(group[train_end + val_size :])
    return splits


class SquarePad:
    def __call__(self, image: Image.Image) -> Image.Image:
        side = max(image.size)
        return ImageOps.pad(image, (side, side), color=(0, 0, 0))


def image_transform(training: bool) -> transforms.Compose:
    steps: list[object] = [SquarePad(), transforms.Resize((224, 224), antialias=True)]
    if training:
        steps += [
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


class CardDataset(Dataset):
    def __init__(
        self, samples: list[tuple[Path, str]], classes: list[str], training: bool = False
    ) -> None:
        self.samples = samples
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = image_transform(training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[label]


def class_counts(samples: list[tuple[Path, str]]) -> Counter[str]:
    return Counter(label for _, label in samples)

