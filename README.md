# TCGScan

A small PyTorch/ResNet-18 tool that learns to classify Pokemon card expansion sets.

## Setup

Download the [Pokemon TCG image dataset](https://www.kaggle.com/datasets/ellimaaac/pokemon-tcg-all-image-cards?resource=download)
and extract it below `data/`. Nested archive directories are discovered automatically.

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run tcgscan inspect --data data/archive
```

Or with a conventional virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
tcgscan inspect --data data/archive
```

## Train and use the classifier

```bash
tcgscan train --data data/archive --epochs 10
tcgscan evaluate --data data/archive --checkpoint models/best.pt
tcgscan predict path/to/card.jpg --checkpoint models/best.pt
```

Training downloads pretrained ResNet-18 weights on its first run. Pass `--no-pretrained`
to train from random weights, though accuracy will generally be worse. The tool automatically
uses CUDA, Apple MPS, or CPU in that order; override it with `--device` when needed.

The class label is the image's parent-directory name. The checkpoint stores the class list and
split seed so `evaluate` can reproduce the same held-out test set. Exact individual-card lookup
is a future embedding/retrieval step rather than a 20,000-class CNN problem.
