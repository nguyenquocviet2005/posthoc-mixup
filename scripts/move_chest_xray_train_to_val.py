#!/usr/bin/env python3
"""Move a fraction of Chest X-ray train images into val, class-balanced.

Expected layout:
  root/
    train/NORMAL
    train/PNEUMONIA
    val/NORMAL
    val/PNEUMONIA
    test/NORMAL
    test/PNEUMONIA
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: Path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Move part of Chest X-ray train split into val split.")
    parser.add_argument("--root", type=Path, required=True, help="Chest X-ray dataset root containing train/val/test.")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of each train class to move.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to write move manifest JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files.")
    args = parser.parse_args()

    if not 0.0 < args.fraction < 1.0:
        raise ValueError("--fraction must be between 0 and 1")

    root = args.root
    train_root = root / "train"
    val_root = root / "val"
    classes = ["NORMAL", "PNEUMONIA"]

    if not train_root.exists():
        raise FileNotFoundError(f"Missing train directory: {train_root}")
    val_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (root / f"moved_train_to_val_seed{args.seed}_frac{args.fraction:g}.json")
    if manifest_path.exists() and not args.dry_run:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. "
            "Refusing to move again because the split may already have been adjusted."
        )

    rng = random.Random(args.seed)
    manifest = {
        "root": str(root),
        "fraction": args.fraction,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "classes": {},
    }

    for class_name in classes:
        train_dir = train_root / class_name
        val_dir = val_root / class_name
        val_dir.mkdir(parents=True, exist_ok=True)

        files = list_images(train_dir)
        selected = files[:]
        rng.shuffle(selected)
        n_move = math.floor(len(selected) * args.fraction)
        selected = sorted(selected[:n_move])

        class_moves = []
        for src in selected:
            dst = val_dir / src.name
            if dst.exists():
                raise FileExistsError(f"Destination already exists: {dst}")
            class_moves.append({"src": str(src), "dst": str(dst)})
            if not args.dry_run:
                shutil.move(str(src), str(dst))

        manifest["classes"][class_name] = {
            "train_before": len(files),
            "moved": len(class_moves),
            "train_after": len(files) - len(class_moves),
            "moves": class_moves,
        }

    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for class_name, info in manifest["classes"].items():
        print(
            f"{class_name}: moved {info['moved']} "
            f"from train ({info['train_before']} -> {info['train_after']})"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
