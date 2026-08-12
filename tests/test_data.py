from pathlib import Path

from PIL import Image

from tcgscan.data import CardDataset, discover_samples, split_samples


def test_discovery_split_and_loading(tmp_path: Path):
    for label in ("set-a", "set-b"):
        directory = tmp_path / "nested" / label
        directory.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (60, 84), (index, 20, 30)).save(directory / f"{index}.jpg")

    samples, classes = discover_samples(tmp_path)
    splits = split_samples(samples, seed=7)

    assert classes == ["set-a", "set-b"]
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 16,
        "val": 2,
        "test": 2,
    }
    image, label = CardDataset(splits["train"], classes)[0]
    assert image.shape == (3, 224, 224)
    assert label in (0, 1)


def test_tiny_classes_remain_trainable(tmp_path: Path):
    samples = []
    for count in (1, 2, 3):
        samples += [(tmp_path / f"{count}-{index}.jpg", str(count)) for index in range(count)]

    splits = split_samples(samples)
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 1

