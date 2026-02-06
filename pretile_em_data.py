#!/usr/bin/env python3
"""
Pre-tile large EM micrographs into small JPEG tiles using all CPUs.

Why pre-tile?
    Loading a 7676×7420 JPEG to extract one 224×224 tile wastes ~99.9%
    of the decoded pixels.  Pre-tiling converts the I/O bottleneck from
    "decode 56 Mpx per sample" to "decode 50 Kpx per sample" — roughly
    1000× faster per sample.

Output structure:
    /data/em_nepa/train_tiles_224/
        10005_stack_0002_2x_SumCorr_y0000_x0000.jpg
        10005_stack_0002_2x_SumCorr_y0000_x0224.jpg
        ...

Usage:
    # Tile both train and val using 128 workers
    python pretile_em_data.py \
        --source_dir /data/em_nepa/train \
        --output_dir /data/em_nepa/train_tiles_224 \
        --tile_size 224 --overlap 0.0 --num_workers 128 --quality 95

    python pretile_em_data.py \
        --source_dir /data/em_nepa/val \
        --output_dir /data/em_nepa/val_tiles_224 \
        --tile_size 224 --overlap 0.0 --num_workers 128 --quality 95
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


def tile_one_image(args: tuple) -> tuple[str, int]:
    """Tile a single image. Runs in a worker process."""
    src_path, output_dir, tile_size, stride, quality = args

    try:
        img = Image.open(src_path)

        # Handle 16-bit / float images
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(img, dtype=np.float32)
            lo, hi = np.percentile(arr, [0.5, 99.5])
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        elif img.mode not in ("L", "RGB"):
            img = img.convert("L")

        w, h = img.size
        stem = Path(src_path).stem
        count = 0

        if h < tile_size or w < tile_size:
            # Image smaller than tile: save resized
            tile = img.resize((tile_size, tile_size), Image.BICUBIC)
            out_name = f"{stem}_y0000_x0000.jpg"
            tile.save(os.path.join(output_dir, out_name), "JPEG", quality=quality)
            return src_path, 1

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                out_name = f"{stem}_y{y:04d}_x{x:04d}.jpg"
                tile.save(os.path.join(output_dir, out_name), "JPEG", quality=quality)
                count += 1

        return src_path, count

    except Exception as e:
        print(f"ERROR: {src_path}: {e}", file=sys.stderr)
        return src_path, 0


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tile EM images into small JPEGs for fast data loading.",
    )
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory with source images (e.g. /data/em_nepa/train).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for tiles.")
    parser.add_argument("--tile_size", type=int, default=224,
                        help="Tile edge length in pixels (default: 224).")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Overlap ratio between tiles 0.0–0.9 (default: 0.0).")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of parallel processes (default: 64).")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG save quality 1–100 (default: 95).")
    args = parser.parse_args()

    # ---- Scan source images ----
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    source = Path(args.source_dir)
    image_paths = sorted([
        str(p) for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ])
    print(f"Found {len(image_paths)} images in {args.source_dir}")

    if not image_paths:
        print("No images found. Exiting.")
        return

    # ---- Prepare output ----
    os.makedirs(args.output_dir, exist_ok=True)

    stride = max(1, int(args.tile_size * (1.0 - args.overlap)))
    print(f"Tile size: {args.tile_size}, stride: {stride}, overlap: {args.overlap}")
    print(f"JPEG quality: {args.quality}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print()

    # ---- Estimate ----
    sample_img = Image.open(image_paths[0])
    sw, sh = sample_img.size
    tiles_per_sample = ((sh - args.tile_size) // stride + 1) * ((sw - args.tile_size) // stride + 1)
    est_total = tiles_per_sample * len(image_paths)
    print(f"Sample image: {sw}×{sh} → ~{tiles_per_sample} tiles/image")
    print(f"Estimated total tiles: ~{est_total:,}")
    print()

    # ---- Parallel tiling ----
    work_items = [
        (path, args.output_dir, args.tile_size, stride, args.quality)
        for path in image_paths
    ]

    t0 = time.time()
    total_tiles = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(tile_one_image, item): item[0] for item in work_items}

        for future in as_completed(futures):
            src_path, count = future.result()
            total_tiles += count
            done += 1

            if done % 100 == 0 or done == len(image_paths):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(image_paths) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done:>5}/{len(image_paths)}]  "
                    f"tiles: {total_tiles:>10,}  "
                    f"rate: {rate:.1f} img/s  "
                    f"ETA: {eta:.0f}s"
                )

    elapsed = time.time() - t0
    print()
    print(f"Done!  {total_tiles:,} tiles from {len(image_paths)} images in {elapsed:.1f}s")
    print(f"Output: {args.output_dir}")

    # ---- Summary ----
    out_files = list(Path(args.output_dir).glob("*.jpg"))
    total_bytes = sum(f.stat().st_size for f in out_files)
    print(f"Disk usage: {total_bytes / 1e9:.2f} GB  ({len(out_files)} files)")


if __name__ == "__main__":
    main()
