from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CardDataset, discover_samples, split_samples
from .model import choose_device, create_model


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_top5 = total_items = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        total_top5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
        total_items += labels.size(0)
    return total_loss / total_items, total_correct / total_items, total_top5 / total_items


def make_loader(samples, classes, batch_size, workers, training=False):
    return DataLoader(
        CardDataset(samples, classes, training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(
    data: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    workers: int,
    seed: int,
    device_name: str,
    pretrained: bool,
) -> None:
    torch.manual_seed(seed)
    samples, classes = discover_samples(data)
    splits = split_samples(samples, seed)
    device = choose_device(device_name)
    print(
        f"device={device} classes={len(classes)} "
        + " ".join(f"{name}={len(items)}" for name, items in splits.items())
    )

    train_loader = make_loader(splits["train"], classes, batch_size, workers, True)
    val_loader = make_loader(splits["val"], classes, batch_size, workers)
    model = create_model(len(classes), pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        train_loss, train_accuracy, train_top5 = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_accuracy, val_top5 = run_epoch(model, val_loader, device)
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.1%} train_top5={train_top5:.1%} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.1%} "
            f"val_top5={val_top5:.1%} seconds={time.monotonic() - started:.0f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "seed": seed,
                    "architecture": "resnet18",
                },
                output,
            )
            print(f"saved={output}")


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Checkpoint does not contain a supported ResNet-18 model")
    model = create_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def evaluate(
    data: Path, checkpoint_path: Path, batch_size: int, workers: int, device_name: str
) -> None:
    device = choose_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    samples, found_classes = discover_samples(data)
    if found_classes != checkpoint["classes"]:
        raise ValueError("Dataset classes differ from the classes stored in the checkpoint")
    test_samples = split_samples(samples, checkpoint["seed"])["test"]
    loss, accuracy, top5 = run_epoch(
        model, make_loader(test_samples, found_classes, batch_size, workers), device
    )
    print(
        f"split=test images={len(test_samples)} loss={loss:.4f} "
        f"accuracy={accuracy:.1%} top5={top5:.1%}"
    )
