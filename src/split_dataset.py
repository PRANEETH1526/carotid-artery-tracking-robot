"""Split preprocessed carotid ultrasound dataset into train/val/test sets by patient/session.

Splitting Strategy:
  - 70% Train / 15% Validation / 15% Test
  - Grouped by session ID (extracted from raw filename) to prevent data leakage.
  - Verifies 1-to-1 matching between images and masks across all splits.
  - Optional '--augment' flag applies geometric and photometric augmentations to the TRAIN set only.
"""

import os
import re
import csv
import glob
import random
import shutil
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def get_session_id_from_raw(raw_name: str) -> str:
    """Extract patient/session ID from raw ultrasound image filename."""
    m = re.match(r"^(.*)_slice_\d+\.png$", raw_name)
    if m:
        return m.group(1)
    return os.path.splitext(raw_name)[0]


def augment_pair(
    img: np.ndarray, mask: np.ndarray, aug_type: str, angle: float = 0.0, factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply synchronized augmentation to an image-mask pair.

    Args:
        img: Preprocessed image (uint8 256x256).
        mask: Preprocessed binary mask (uint8 256x256, {0, 255}).
        aug_type: Type of augmentation: 'rotate', 'hflip', or 'brightness'.
        angle: Rotation angle in degrees (for 'rotate').
        factor: Brightness multiplier (for 'brightness').

    Returns:
        Tuple of (augmented_image, augmented_mask).
    """
    h, w = img.shape[:2]

    if aug_type == "rotate":
        # Rotation around center
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Image uses linear interpolation; mask MUST use nearest neighbor
        aug_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        aug_mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Ensure mask remains strictly binary {0, 255}
        _, aug_mask = cv2.threshold(aug_mask, 127, 255, cv2.THRESH_BINARY)
        return aug_img, aug_mask

    elif aug_type == "hflip":
        # Horizontal flip
        aug_img = cv2.flip(img, 1)
        aug_mask = cv2.flip(mask, 1)
        return aug_img, aug_mask

    elif aug_type == "brightness":
        # Adjust brightness on image only; mask is invariant to illumination
        aug_img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return aug_img, mask.copy()

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def split_dataset(
    processed_all_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 123,
    augment_train: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Split preprocessed dataset into train, val, and test subsets.

    Args:
        processed_all_dir: Directory containing preprocessed 'images', 'masks', and 'metadata.csv'.
        output_dir: Root output directory where 'train', 'val', and 'test' folders are created.
        train_ratio: Fraction of sessions for training (default: 0.70).
        val_ratio: Fraction of sessions for validation (default: 0.15).
        test_ratio: Fraction of sessions for testing (default: 0.15).
        seed: Random seed for reproducibility (default: 123).
        augment_train: Whether to augment the training set.

    Returns:
        Summary dict containing counts of sessions and images for each split.
    """
    metadata_csv = os.path.join(processed_all_dir, "metadata.csv")
    images_dir = os.path.join(processed_all_dir, "images")
    masks_dir = os.path.join(processed_all_dir, "masks")

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}. Please run preprocess.py first.")

    # Read metadata
    records = []
    with open(metadata_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} records from {metadata_csv}")

    # Group records by session_id
    sessions: Dict[str, List[Dict[str, str]]] = {}
    for r in records:
        sid = r["session_id"]
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(r)

    unique_sessions = sorted(list(sessions.keys()))
    n_sessions = len(unique_sessions)
    print(f"Identified {n_sessions} unique patient/recording sessions.")

    # Shuffle sessions with fixed seed
    random.seed(seed)
    shuffled_sessions = unique_sessions.copy()
    random.shuffle(shuffled_sessions)

    # Calculate session counts per split
    n_train = int(round(n_sessions * train_ratio))
    n_val = int(round(n_sessions * val_ratio))
    # Remainder goes to test
    n_test = n_sessions - n_train - n_val

    # Ensure no split is empty
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(f"Split resulted in an empty set: train={n_train}, val={n_val}, test={n_test}")

    train_sessions = set(shuffled_sessions[:n_train])
    val_sessions = set(shuffled_sessions[n_train:n_train + n_val])
    test_sessions = set(shuffled_sessions[n_train + n_val:])

    # Verify zero session leakage
    assert len(train_sessions & val_sessions) == 0, "Leakage detected between train and val!"
    assert len(train_sessions & test_sessions) == 0, "Leakage detected between train and test!"
    assert len(val_sessions & test_sessions) == 0, "Leakage detected between val and test!"

    splits = {
        "train": (train_sessions, os.path.join(output_dir, "train")),
        "val": (val_sessions, os.path.join(output_dir, "val")),
        "test": (test_sessions, os.path.join(output_dir, "test")),
    }

    summary: Dict[str, Dict[str, int]] = {}

    for split_name, (session_set, split_path) in splits.items():
        out_img_dir = os.path.join(split_path, "images")
        out_mask_dir = os.path.join(split_path, "masks")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # Clear any existing files in split directories
        for f in glob.glob(os.path.join(out_img_dir, "*.png")):
            os.remove(f)
        for f in glob.glob(os.path.join(out_mask_dir, "*.png")):
            os.remove(f)

        split_records = [r for r in records if r["session_id"] in session_set]
        copied_count = 0
        augmented_count = 0

        split_csv_path = os.path.join(split_path, f"{split_name}_metadata.csv")
        split_meta_rows = []

        for r in split_records:
            src_img = os.path.join(images_dir, r["image_filename"])
            src_mask = os.path.join(masks_dir, r["mask_filename"])

            if not os.path.exists(src_img) or not os.path.exists(src_mask):
                print(f"Warning: Missing source pair {src_img} / {src_mask}")
                continue

            dest_img = os.path.join(out_img_dir, r["image_filename"])
            dest_mask = os.path.join(out_mask_dir, r["mask_filename"])

            row_copy = dict(r)
            row_copy["is_augmented"] = "none"
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_mask, dest_mask)
            copied_count += 1
            split_meta_rows.append(row_copy)

            # Optional Augmentation (TRAIN only)
            if augment_train and split_name == "train":
                img_data = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
                mask_data = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)

                stem = os.path.splitext(r["image_filename"])[0]
                mask_stem = os.path.splitext(r["mask_filename"])[0]

                augmentations = [
                    ("rot_p10", "rotate", 10.0, 1.0),
                    ("rot_m10", "rotate", -10.0, 1.0),
                    ("hflip", "hflip", 0.0, 1.0),
                    ("bright_p15", "brightness", 0.0, 1.15),
                    ("bright_m15", "brightness", 0.0, 0.85),
                ]

                for aug_name, aug_mode, ang, fact in augmentations:
                    aug_i, aug_m = augment_pair(img_data, mask_data, aug_mode, angle=ang, factor=fact)
                    aug_img_name = f"{stem}_{aug_name}.png"
                    aug_mask_name = f"{mask_stem}_{aug_name}.png"

                    cv2.imwrite(os.path.join(out_img_dir, aug_img_name), aug_i)
                    cv2.imwrite(os.path.join(out_mask_dir, aug_mask_name), aug_m)
                    augmented_count += 1

                    aug_row = dict(r)
                    aug_row["image_filename"] = aug_img_name
                    aug_row["mask_filename"] = aug_mask_name
                    aug_row["is_augmented"] = aug_name
                    split_meta_rows.append(aug_row)

        # Save split-specific metadata
        if split_meta_rows:
            with open(split_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=split_meta_rows[0].keys())
                writer.writeheader()
                writer.writerows(split_meta_rows)

        # Verify 1-to-1 matching in target directory
        final_imgs = set(os.listdir(out_img_dir))
        final_masks = set(os.listdir(out_mask_dir))

        # Check that img_XXXX.png matches mask_XXXX.png
        unmatched = 0
        for fi in final_imgs:
            matching_mask = fi.replace("img_", "mask_")
            if matching_mask not in final_masks:
                unmatched += 1

        summary[split_name] = {
            "sessions": len(session_set),
            "original_images": copied_count,
            "augmented_images": augmented_count,
            "total_images": len(final_imgs),
            "total_masks": len(final_masks),
            "unmatched_pairs": unmatched,
        }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset by patient session with fixed seed.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/processed/all",
        help="Path to preprocessed all/ folder containing images, masks, and metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/processed",
        help="Root output directory for train/val/test splits.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for splitting (default: 1).")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation to TRAIN split only.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15).")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15).")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    def resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    in_dir = resolve_path(args.input_dir)
    out_dir = resolve_path(args.output_dir)

    print("==================================================")
    print(" CAROTID ULTRASOUND DATASET SPLITTER (SESSION-WISE)")
    print("==================================================")
    print(f"Input Directory:  {in_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Random Seed:      {args.seed}")
    print(f"Augment Train:    {args.augment}")
    print(f"Ratios:           Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print("--------------------------------------------------")

    summary = split_dataset(
        processed_all_dir=in_dir,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        augment_train=args.augment,
    )

    print("\nSplit Results Breakdown:")
    for split_name, counts in summary.items():
        print(f"\n[{split_name.upper()}]")
        print(f"  Sessions:         {counts['sessions']}")
        print(f"  Original Images:  {counts['original_images']}")
        if counts["augmented_images"] > 0:
            print(f"  Augmented Images: {counts['augmented_images']}")
        print(f"  Total Images:     {counts['total_images']}")
        print(f"  Total Masks:      {counts['total_masks']}")
        print(f"  Unmatched Pairs:  {counts['unmatched_pairs']} {'[OK]' if counts['unmatched_pairs'] == 0 else '[ERROR]'}")

    print("\nDataset split complete! No patient sessions cross between train, val, and test.")
