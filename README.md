# TCGScan

TCGScan is a small Python and PyTorch tool for training a CNN to recognize the
expansion set of a Pokemon trading card. It uses transfer learning with a
pretrained ResNet-18 and automatically supports NVIDIA CUDA, Apple MPS, and CPU
execution.

The project is intentionally compact: dataset loading, training, evaluation, and
single-image prediction are all available through one command-line interface.

## Quick start

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run tcgscan inspect --data data/archive
uv run tcgscan train --data data/archive --epochs 10
uv run tcgscan evaluate --data data/archive
uv run tcgscan predict path/to/card.jpg
```

The best model is saved to `models/best.pt` by default.

## Requirements

- Python 3.11 or newer
- Approximately 3.4 GB of disk space for the current image dataset
- An internet connection on the first training run to download pretrained
  ResNet-18 weights
- Optional: an NVIDIA GPU or Apple Silicon Mac for faster training

The core dependencies are PyTorch, TorchVision, and Pillow. Development tools
such as pytest and Ruff are kept in a separate dependency group.

## Installation

### Using uv

```bash
uv sync
```

Use `uv run` before each command so it runs inside the managed environment:

```bash
uv run tcgscan --help
```

### Using venv and pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
tcgscan --help
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

## Dataset setup

Download the [Pokemon TCG image dataset](https://www.kaggle.com/datasets/ellimaaac/pokemon-tcg-all-image-cards?resource=download)
and extract it under `data/`. The included dataset currently contains 20,983
images across 195 expansion directories.

TCGScan searches recursively, so extra archive directories are harmless. Each
directory containing images becomes one classification label:

```text
data/archive/
├── Pokemon TCG/
│   └── Pokemon TCG/
│       ├── evolving-skies/
│       │   ├── card-001.jpg
│       │   └── card-002.jpg
│       └── lost-thunder/
│           └── card-001.jpg
└── chaos-rising/
    └── chaos-rising/
        └── card-001.jpg
```

Supported image formats are JPEG, PNG, and WebP. The immediate parent directory
of each image is used as its expansion label.

Before training, verify what TCGScan found:

```bash
tcgscan inspect --data data/archive
```

The first line reports the total image and class counts, followed by the number
of images in every class.

## Training

Start a standard training run with:

```bash
tcgscan train --data data/archive --epochs 10
```

TCGScan creates a deterministic per-class 80/10/10 training, validation, and
test split. Very small classes retain a training sample even when there are not
enough images for every split.

During training, the tool reports:

- Training and validation loss
- Top-1 accuracy
- Top-5 accuracy
- Time per epoch
- When a new best checkpoint is saved

Images are padded to a square without cropping away card details and resized to
224 by 224 pixels. Training augmentation simulates modest camera perspective,
rotation, lighting variation, and blur. Cards are not horizontally flipped
because mirrored text is not representative of real input.

Useful training options:

```bash
tcgscan train \
  --data data/archive \
  --output models/best.pt \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.0003 \
  --workers 0 \
  --seed 42 \
  --device auto
```

Use `--no-pretrained` to initialize ResNet-18 with random weights. This avoids
the initial weight download but will generally require more training data and
time.

### Device selection

The default `--device auto` checks for acceleration in this order:

1. NVIDIA CUDA
2. Apple MPS
3. CPU

You can force a device when troubleshooting:

```bash
tcgscan train --data data/archive --device cpu
```

If GPU memory is exhausted, reduce `--batch-size`, for example to `16` or `8`.
Increasing `--workers` can improve data-loading speed on some systems, while the
default of `0` is the most portable setting.

## Evaluation

Evaluate the best checkpoint on the held-out test split:

```bash
tcgscan evaluate \
  --data data/archive \
  --checkpoint models/best.pt
```

The checkpoint stores the class names and random seed used during training, so
evaluation reconstructs the same test split. Evaluation stops with an error if
the dataset's class list differs from the checkpoint's class list.

## Predicting an image

Classify a single card photograph or scan:

```bash
tcgscan predict path/to/card.jpg --checkpoint models/best.pt
```

By default, the five most likely expansion sets are printed with their
probabilities:

```text
 72.14%  evolving-skies
 14.82%  chilling-reign
  6.03%  fusion-strike
  4.11%  brilliant-stars
  1.37%  lost-origin
```

Change the number of results with `--top-k`:

```bash
tcgscan predict path/to/card.jpg --top-k 3
```

## Command reference

```text
tcgscan inspect   Summarize discovered images and expansion classes
tcgscan train     Train the expansion classifier and save its best checkpoint
tcgscan evaluate  Measure checkpoint accuracy on the held-out test split
tcgscan predict   Print the most likely expansions for one image
```

Run `tcgscan <command> --help` for every available option.

## Development

Install the development dependencies and run the checks with uv:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
```

Project layout:

```text
src/tcgscan/
├── cli.py       Command-line interface
├── data.py      Discovery, splitting, augmentation, and dataset loading
├── engine.py    Training, checkpointing, and evaluation
└── model.py     ResNet-18 construction and device selection
tests/
└── test_data.py
```

## Current scope and limitations

TCGScan currently predicts an expansion set, not the exact card identity. Most
cards in the reference dataset have only one image, which is not enough to train
a conventional 20,000-class classifier reliably.

Exact card recognition is better handled as a later retrieval stage: use the
CNN to generate an embedding, search for visually similar reference cards, and
optionally confirm the match using OCR on the card name and collector number.
Real camera photos may also differ from the clean reference scans because of
background clutter, sleeves, glare, shadows, and partial occlusion. The existing
augmentation provides a baseline, but representative phone-camera training
images would improve real-world accuracy.
