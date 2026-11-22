"""Preprocess ultrasound images and binary lumen masks for carotid artery tracking.

Pipeline:
  Image:  Grayscale -> Resize(256, 256) -> Median Blur(k=5) -> CLAHE(clip=2.0, tile=8x8) -> Float32 [0.0, 1.0]
  Mask:   Resize(256, 256, Nearest) -> Strictly binary {0, 255} uint8
"""

import os
import re
import csv
import glob
import argparse
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
from tqdm import tqdm


def preprocess(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess a raw ultrasound image.

    Args:
        img: Input image as numpy array (grayscale HxW or color HxWxC BGR).
        size: Target (width, height) resolution. Default is (256, 256).

    Returns:
        Preprocessed image as float32 numpy array with values in [0.0, 1.0].
    """
    if img is None:
        raise ValueError("Input image is None. Please provide a valid image array.")

    # 1. Convert to grayscale if 3 channels
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Resize to target size (using bilinear interpolation for smooth downsampling)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)

    # 3. Median filter (kernel size 5) to remove speckle noise common in ultrasound
    denoised = cv2.medianBlur(resized, 5)

    # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Enhances acoustic contrast between vessel lumen and arterial wall
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Normalize pixel values to float32 range [0.0, 1.0]
    normalized = enhanced.astype(np.float32) / 255.0

    return normalized


def preprocess_mask(mask: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess a raw binary ground-truth mask.

    Args:
        mask: Input mask as numpy array (HxW or HxWxC).
        size: Target (width, height) resolution. Default is (256, 256).

    Returns:
        Binary mask as uint8 numpy array with values strictly {0, 255}.
    """
    if mask is None:
        raise ValueError("Input mask is None. Please provide a valid mask array.")

    # If 3 channels, take single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Resize using INTER_NEAREST to prevent fuzzy/interpolated boundary values
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # Strict binary thresholding to guarantee only 0 or 255
    # Any non-zero pixel in ground truth becomes 255 (artery lumen), background becomes 0
    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    return binary_mask.astype(np.uint8)


def extract_session_id(filename: str) -> str:
    """Extract patient/session ID from filename.

    Example: '202201121748100022VAS_slice_1069.png' -> '202201121748100022VAS'
    """
    base = os.path.basename(filename)
    match = re.match(r"^(.*)_slice_\d+\.png$", base)
    if match:
        return match.group(1)
    # Fallback to file root if pattern differs
    return os.path.splitext(base)[0]


def batch_preprocess(
    raw_images_dir: str,
    raw_masks_dir: str,
    output_images_dir: str,
    output_masks_dir: str,
    metadata_csv_path: Optional[str] = None,
    size: Tuple[int, int] = (256, 256),
) -> List[Dict[str, Any]]:
    """Batch processes all raw image and mask pairs into target directories.

    Generates matched filenames (img_0001.png <-> mask_0001.png) and records
    metadata (original name, session ID, sizes).

    Args:
        raw_images_dir: Directory containing raw ultrasound images.
        raw_masks_dir: Directory containing raw binary masks.
        output_images_dir: Destination directory for preprocessed images.
        output_masks_dir: Destination directory for preprocessed masks.
        metadata_csv_path: Optional path to save CSV recording file mappings.
        size: Target (width, height) image resolution.

    Returns:
        List of metadata dictionaries for all processed pairs.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(raw_images_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(raw_masks_dir, "*.png")))

    mask_dict = {os.path.basename(f): f for f in mask_files}

    matched_pairs = []
    unmatched_images = []

    for img_path in image_files:
        fname = os.path.basename(img_path)
        if fname in mask_dict:
            matched_pairs.append((img_path, mask_dict[fname]))
        else:
            unmatched_images.append(fname)

    if unmatched_images:
        print(f"Warning: {len(unmatched_images)} images had no matching mask!")

    print(f"Found {len(matched_pairs)} matched image/mask pairs to process.")

    metadata_records: List[Dict[str, Any]] = []

    for idx, (img_path, mask_path) in enumerate(tqdm(matched_pairs, desc="Preprocessing")):
        # Read raw files
        raw_img = cv2.imread(img_path)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if raw_img is None or raw_mask is None:
            print(f"Skipping corrupted file pair: {img_path}")
            continue

        orig_fname = os.path.basename(img_path)
        session_id = extract_session_id(orig_fname)

        # Preprocess
        proc_img_float = preprocess(raw_img, size=size)
        proc_mask = preprocess_mask(raw_mask, size=size)

        # Scale float32 [0, 1] to uint8 [0, 255] for saving as standard PNG
        proc_img_uint8 = (proc_img_float * 255.0).round().astype(np.uint8)

        # Matched naming convention: img_0001.png <-> mask_0001.png (4 digits for up to 9999 files)
        new_name_base = f"{idx + 1:04d}"
        img_out_name = f"img_{new_name_base}.png"
        mask_out_name = f"mask_{new_name_base}.png"

        cv2.imwrite(os.path.join(output_images_dir, img_out_name), proc_img_uint8)
        cv2.imwrite(os.path.join(output_masks_dir, mask_out_name), proc_mask)

        metadata_records.append({
            "index": idx + 1,
            "image_filename": img_out_name,
            "mask_filename": mask_out_name,
            "original_filename": orig_fname,
            "session_id": session_id,
            "raw_height": raw_img.shape[0],
            "raw_width": raw_img.shape[1],
            "proc_height": size[1],
            "proc_width": size[0],
            "lumen_pixels": int(np.count_nonzero(proc_mask == 255)),
        })

    # Save metadata CSV if path provided
    if metadata_csv_path and metadata_records:
        os.makedirs(os.path.dirname(os.path.abspath(metadata_csv_path)), exist_ok=True)
        keys = metadata_records[0].keys()
        with open(metadata_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_records)
        print(f"Metadata saved to: {metadata_csv_path}")

    return metadata_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ultrasound images and masks.")
    parser.add_argument(
        "--raw-img-dir",
        type=str,
        default="dataset/raw/images",
        help="Path to raw images folder.",
    )
    parser.add_argument(
        "--raw-mask-dir",
        type=str,
        default="dataset/raw/masks",
        help="Path to raw masks folder.",
    )
    parser.add_argument(
        "--out-img-dir",
        type=str,
        default="dataset/processed/all/images",
        help="Path to output preprocessed images folder.",
    )
    parser.add_argument(
        "--out-mask-dir",
        type=str,
        default="dataset/processed/all/masks",
        help="Path to output preprocessed masks folder.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="dataset/processed/all/metadata.csv",
        help="Path to output metadata CSV.",
    )
    parser.add_argument("--width", type=int, default=256, help="Target width.")
    parser.add_argument("--height", type=int, default=256, help="Target height.")

    args = parser.parse_args()

    # Determine paths relative to script location if default relative paths given
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    def resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    raw_imgs = resolve_path(args.raw_img_dir)
    raw_msks = resolve_path(args.raw_mask_dir)
    out_imgs = resolve_path(args.out_img_dir)
    out_msks = resolve_path(args.out_mask_dir)
    meta_csv = resolve_path(args.metadata_csv)

    print("=== Batch Preprocessing Starting ===")
    print(f"Raw Images Dir: {raw_imgs}")
    print(f"Raw Masks Dir:  {raw_msks}")
    print(f"Output Images:  {out_imgs}")
    print(f"Output Masks:   {out_msks}")

    records = batch_preprocess(
        raw_images_dir=raw_imgs,
        raw_masks_dir=raw_msks,
        output_images_dir=out_imgs,
        output_masks_dir=out_msks,
        metadata_csv_path=meta_csv,
        size=(args.width, args.height),
    )
    print(f"Successfully preprocessed {len(records)} images and masks to {args.width}x{args.height}!")
