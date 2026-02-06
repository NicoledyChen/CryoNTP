#!/usr/bin/env python3
"""
Prepare cryo-EM train / val split from cryoppp_lite.

Creates a directory structure with symlinks:

    /data/em_nepa/
    ├── train/          (symlinks to ~90% of micrographs)
    └── val/            (symlinks to ~10% of micrographs)

Two split strategies (--split_mode):
  - "image"   : within each dataset, randomly hold out images  (default)
  - "dataset" : hold out entire datasets for validation

Usage:
    python prepare_em_data.py                          # default 90/10 image split
    python prepare_em_data.py --val_ratio 0.15         # 85/15 split
    python prepare_em_data.py --split_mode dataset     # hold out whole datasets
"""

import argparse
import os
import random
from pathlib import Path


def scan_datasets(source_dir: str) -> dict[str, list[str]]:
    """Return {dataset_id: [list of image paths]} for all datasets."""
    datasets = {}
    source = Path(source_dir)

    for ds_dir in sorted(source.iterdir()):
        if not ds_dir.is_dir():
            continue
        mic_dir = ds_dir / "micrographs"
        if not mic_dir.is_dir():
            continue

        images = sorted([
            str(p) for p in mic_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        ])
        if images:
            datasets[ds_dir.name] = images

    return datasets


def split_by_image(
    datasets: dict[str, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Within each dataset, randomly assign images to train or val."""
    rng = random.Random(seed)
    train_paths, val_paths = [], []

    for ds_id in sorted(datasets.keys()):
        images = list(datasets[ds_id])
        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio))
        val_paths.extend(images[:n_val])
        train_paths.extend(images[n_val:])

    return train_paths, val_paths


def split_by_dataset(
    datasets: dict[str, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Hold out entire datasets for validation."""
    rng = random.Random(seed)
    ds_ids = sorted(datasets.keys())
    rng.shuffle(ds_ids)

    n_val = max(1, int(len(ds_ids) * val_ratio))
    val_ds = set(ds_ids[:n_val])

    train_paths, val_paths = [], []
    for ds_id in sorted(datasets.keys()):
        if ds_id in val_ds:
            val_paths.extend(datasets[ds_id])
        else:
            train_paths.extend(datasets[ds_id])

    return train_paths, val_paths


def create_symlinks(paths: list[str], target_dir: str):
    """Create uniquely-named symlinks in target_dir pointing to source images."""
    os.makedirs(target_dir, exist_ok=True)

    for src_path in paths:
        src = Path(src_path)
        # Unique name: {dataset_id}_{filename}
        parts = src.parts
        # Find the dataset ID (parent of micrographs/)
        mic_idx = parts.index("micrographs")
        ds_id = parts[mic_idx - 1]
        link_name = f"{ds_id}_{src.name}"
        link_path = os.path.join(target_dir, link_name)

        if os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(src_path, link_path)


def main():
    parser = argparse.ArgumentParser(description="Build train/val split for EM pretraining.")
    parser.add_argument(
        "--source_dir", type=str, default="/data/cryoppp_lite",
        help="Root directory of cryoppp_lite.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/data/em_nepa",
        help="Output directory for train/ and val/ symlinks.",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.10,
        help="Fraction of data for validation (default: 0.10).",
    )
    parser.add_argument(
        "--split_mode", type=str, default="image", choices=["image", "dataset"],
        help="'image': split within each dataset. 'dataset': hold out whole datasets.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # ---- Scan ----
    print(f"Scanning {args.source_dir} ...")
    datasets = scan_datasets(args.source_dir)
    total_images = sum(len(v) for v in datasets.values())
    print(f"  Found {len(datasets)} datasets, {total_images} images total.")
    print()

    # ---- Split ----
    if args.split_mode == "image":
        print(f"Splitting by IMAGE (val_ratio={args.val_ratio}, seed={args.seed})")
        train_paths, val_paths = split_by_image(datasets, args.val_ratio, args.seed)
    else:
        print(f"Splitting by DATASET (val_ratio={args.val_ratio}, seed={args.seed})")
        train_paths, val_paths = split_by_dataset(datasets, args.val_ratio, args.seed)

    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    print()

    # ---- Per-dataset breakdown ----
    train_set = set(train_paths)
    val_set = set(val_paths)

    print(f"{'Dataset':>8}  {'Train':>6}  {'Val':>5}  {'Total':>6}  {'Size (sample)'}")
    print("-" * 65)

    for ds_id in sorted(datasets.keys()):
        images = datasets[ds_id]
        n_train = sum(1 for p in images if p in train_set)
        n_val = sum(1 for p in images if p in val_set)

        # Get image size from first image
        from PIL import Image
        sample = Image.open(images[0])
        w, h = sample.size

        print(f"{ds_id:>8}  {n_train:>6}  {n_val:>5}  {len(images):>6}  {w}x{h}")

    print("-" * 65)
    print(f"{'TOTAL':>8}  {len(train_paths):>6}  {len(val_paths):>5}  {total_images:>6}")
    print()

    # ---- Create symlinks ----
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")

    print(f"Creating symlinks ...")
    print(f"  Train → {train_dir}")
    create_symlinks(train_paths, train_dir)
    print(f"  Val   → {val_dir}")
    create_symlinks(val_paths, val_dir)

    print()
    print("Done!")
    print()
    print("You can now run NEPA-EM pretraining:")
    print(f"  --image_dir {train_dir}")
    print(f"  --val_image_dir {val_dir}")


if __name__ == "__main__":
    main()
