from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .data import class_counts, discover_samples, image_transform
from .engine import evaluate, load_checkpoint, train
from .model import choose_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tcgscan", description="Train and use a card CNN")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="summarize a card image dataset")
    inspect.add_argument("--data", type=Path, default=Path("data/archive"))

    training = commands.add_parser("train", help="train an expansion classifier")
    training.add_argument("--data", type=Path, default=Path("data/archive"))
    training.add_argument("--output", type=Path, default=Path("models/best.pt"))
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    training.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    evaluation = commands.add_parser("evaluate", help="evaluate the saved model")
    evaluation.add_argument("--data", type=Path, default=Path("data/archive"))
    evaluation.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--workers", type=int, default=0)
    evaluation.add_argument("--device", default="auto")

    predict = commands.add_parser("predict", help="classify one card image")
    predict.add_argument("image", type=Path)
    predict.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        samples, classes = discover_samples(args.data)
        counts = class_counts(samples)
        print(f"images={len(samples)} classes={len(classes)}")
        for name in classes:
            print(f"{counts[name]:4}  {name}")
    elif args.command == "train":
        train(
            args.data,
            args.output,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.workers,
            args.seed,
            args.device,
            args.pretrained,
        )
    elif args.command == "evaluate":
        evaluate(args.data, args.checkpoint, args.batch_size, args.workers, args.device)
    else:
        device = choose_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        with Image.open(args.image) as image:
            tensor = image_transform(False)(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(1)[0]
        count = min(args.top_k, len(checkpoint["classes"]))
        values, indices = probabilities.topk(count)
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            print(f"{value:7.2%}  {checkpoint['classes'][index]}")


if __name__ == "__main__":
    main()
