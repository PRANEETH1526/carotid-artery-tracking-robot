"""Visualization utilities for carotid ultrasound preprocessing and dataset distribution.

Generates presentation-ready figures saved directly to ./reports/:
  1. preprocessing_comparison.png: 5 samples comparing Raw -> Median -> CLAHE
  2. sample_overlays.png: 5 samples showing Preprocessed Image + Lumen Mask Overlays
  3. dataset_split_counts.png: Bar chart of train/val/test sessions and image counts
"""

import os
import glob
import random
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration for clean, presentation-ready academic figures
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_preprocessing_stages(
    raw_img_dir: str,
    output_path: str,
    n_samples: int = 5,
    seed: int = 42,
) -> None:
    """Generate and save multi-stage comparison for 5 samples:
    Raw -> Denoised (Median k=5) -> CLAHE Enhanced.
    """
    raw_files = sorted(glob.glob(os.path.join(raw_img_dir, "*.png")))
    if not raw_files:
        raise FileNotFoundError(f"No PNG images found in {raw_img_dir}")

    random.seed(seed)
    chosen_files = random.sample(raw_files, min(n_samples, len(raw_files)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Raw Input (749x709)",
        "2. Grayscale (256x256)",
        "3. Median Denoised (k=5)",
        "4. CLAHE Enhanced (clip=2.0)",
    ]

    for row_idx, fpath in enumerate(chosen_files):
        fname = os.path.basename(fpath)
        img_bgr = cv2.imread(fpath)

        # Stage 1: Raw
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Stage 2: Grayscale & resize
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Stage 3: Median blur
        denoised = cv2.medianBlur(resized, 5)

        # Stage 4: CLAHE
        enhanced = clahe.apply(denoised)

        stages = [img_rgb, resized, denoised, enhanced]

        for col_idx, stage_data in enumerate(stages):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ax.imshow(stage_data)
            else:
                ax.imshow(stage_data, cmap="gray", vmin=0, vmax=255)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)

            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_idx, 0].set_ylabel(f"Sample {row_idx + 1}\n{fname[:22]}...", fontsize=9, labelpad=8)

    plt.suptitle("Carotid Ultrasound Preprocessing Pipeline (Speckle Denoising & CLAHE)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved preprocessing comparison figure to: {output_path}")


def plot_sample_overlays(
    images_dir: str,
    masks_dir: str,
    output_path: str,
    n_samples: int = 5,
    seed: int = 42,
) -> None:
    """Generate and save side-by-side figures showing Preprocessed Image,
    Ground Truth Mask, and Green Contour Overlay on the Artery Lumen.
    """
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not img_files:
        raise FileNotFoundError(f"No preprocessed images in {images_dir}")

    random.seed(seed)
    chosen_imgs = random.sample(img_files, min(n_samples, len(img_files)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(11, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Preprocessed Ultrasound (256x256)",
        "Ground Truth Lumen Mask (0/255)",
        "Artery Lumen Overlay",
    ]

    for row_idx, img_p in enumerate(chosen_imgs):
        fname = os.path.basename(img_p)
        mask_name = fname.replace("img_", "mask_")
        mask_p = os.path.join(masks_dir, mask_name)

        if not os.path.exists(mask_p):
            # Try same name
            mask_p = os.path.join(masks_dir, fname)

        img_gray = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        mask_bin = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        if img_gray is None or mask_bin is None:
            continue

        # Strict binary
        _, mask_bin = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)

        # Color overlay
        color_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        green = np.zeros_like(color_img)
        green[:, :, 1] = mask_bin
        overlay = cv2.addWeighted(color_img, 0.70, green, 0.30, 0)

        # Draw green contour around lumen boundary
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        items = [img_gray, mask_bin, overlay_rgb]

        for col_idx, item in enumerate(items):
            ax = axes[row_idx, col_idx]
            if col_idx < 2:
                ax.imshow(item, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(item)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)

            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_idx, 0].set_ylabel(f"Sample {row_idx + 1}\n{fname}", fontsize=9, labelpad=8)

    plt.suptitle("Common Carotid Artery Lumen Ground-Truth Overlays", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved sample overlays figure to: {output_path}")


def plot_dataset_splits(
    processed_dir: str,
    output_path: str,
) -> None:
    """Generate and save publication-quality bar chart showing session and image counts
    across Train, Validation, and Test splits.
    """
    splits = ["train", "val", "test"]
    split_labels = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    img_counts = []
    session_counts = []

    for sp in splits:
        sp_img_dir = os.path.join(processed_dir, sp, "images")
        sp_csv = os.path.join(processed_dir, sp, f"{sp}_metadata.csv")

        # Count images
        imgs = glob.glob(os.path.join(sp_img_dir, "*.png"))
        img_counts.append(len(imgs))

        # Count sessions from metadata or filenames
        if os.path.exists(sp_csv):
            import csv
            sessions = set()
            with open(sp_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sessions.add(r.get("session_id", ""))
            session_counts.append(len(sessions))
        else:
            session_counts.append(0)

    total_imgs = sum(img_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Chart 1: Image counts
    colors_imgs = ["#2b5c8f", "#d97724", "#2e7d32"]
    bars1 = ax1.bar(split_labels, img_counts, color=colors_imgs, width=0.55, edgecolor="#333333", linewidth=1)
    ax1.set_title("Number of Ultrasound Images", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Image Count", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, max(img_counts) * 1.18)

    for bar, count in zip(bars1, img_counts):
        pct = (count / total_imgs * 100.0) if total_imgs > 0 else 0
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 15,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Chart 2: Session counts
    colors_sess = ["#5c768d", "#e09153", "#5c9e60"]
    bars2 = ax2.bar(split_labels, session_counts, color=colors_sess, width=0.55, edgecolor="#333333", linewidth=1)
    ax2.set_title("Number of Independent Recording Sessions", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Session Count", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    total_sess = sum(session_counts)
    ax2.set_ylim(0, max(session_counts) * 1.22)

    for bar, count in zip(bars2, session_counts):
        pct = (count / total_sess * 100.0) if total_sess > 0 else 0
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.6,
            f"{count} sessions\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle("Dataset Split Breakdown (Session-Wise, No Subject Overlap)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset split breakdown chart to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset visualization figures.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/raw/images",
        help="Path to raw ultrasound images.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="dataset/processed",
        help="Path to processed dataset root folder.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Path to reports output folder.",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    def resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    raw_path = resolve_path(args.raw_dir)
    proc_path = resolve_path(args.processed_dir)
    rep_path = resolve_path(args.reports_dir)

    print("=== Generating Visualization Figures ===")
    # 1. Preprocessing comparison
    comp_out = os.path.join(rep_path, "preprocessing_comparison.png")
    plot_preprocessing_stages(raw_path, comp_out, n_samples=5)

    # 2. Sample overlays (use train split if available, else all)
    train_imgs = os.path.join(proc_path, "train", "images")
    train_masks = os.path.join(proc_path, "train", "masks")
    if not os.path.exists(train_imgs):
        train_imgs = os.path.join(proc_path, "all", "images")
        train_masks = os.path.join(proc_path, "all", "masks")

    overlay_out = os.path.join(rep_path, "sample_overlays.png")
    plot_sample_overlays(train_imgs, train_masks, overlay_out, n_samples=5)

    # 3. Split counts bar chart
    split_chart_out = os.path.join(rep_path, "dataset_split_counts.png")
    plot_dataset_splits(proc_path, split_chart_out)

    print("All visualization figures created successfully in reports/!")
