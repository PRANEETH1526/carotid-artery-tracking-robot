"""Training Script for Carotid Artery Lumen U-Net.

Owner: Srujan (Module 1)
Trains from scratch on dataset/processed/train, validates on dataset/processed/val.
Selects best model strictly by VALIDATION Dice score and saves to detection/models/unet_best.pt.
"""

import os
import sys
import csv
import time
import random
import argparse
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import UNet
from dataset import CarotidDataset


# ==========================================
# Loss Functions & Evaluation Metrics
# ==========================================
class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Soft Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> Tuple[float, float]:
    """Calculate binary Dice score and IoU from logits."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        preds_f = preds.contiguous().view(-1)
        targets_f = targets.contiguous().view(-1)

        intersection = (preds_f * targets_f).sum().item()
        total_p = preds_f.sum().item()
        total_g = targets_f.sum().item()
        union = total_p + total_g - intersection

        dice = (2.0 * intersection + smooth) / (total_p + total_g + smooth)
        iou = (intersection + smooth) / (union + smooth)

    return float(dice), float(iou)


# ==========================================
# Main Training Loop
# ==========================================
def run_training(
    data_dir: str = "carotid_project/dataset/processed",
    weights_path: str = "carotid_project/detection/models/unet_best.pt",
    log_csv_path: str = "carotid_project/detection/training_log.csv",
    curves_png_path: str = "carotid_project/reports/detection_training_curves.png",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 6,
    seed: int = 42,
    device_str: str = "auto",
):
    # Set deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 65)
    print("CAROTID ARTERY U-NET TRAINING (Srujan - Module 1)")
    print("=" * 65)
    print(f"Device:                 {device}")
    print(f"Epochs:                 {epochs}")
    print(f"Batch Size:             {batch_size}")
    print(f"Initial Learning Rate:  {lr}")
    print(f"Random Seed:            {seed}")
    print(f"Target Checkpoint:      {weights_path}")
    print(f"Training Log CSV:       {log_csv_path}")
    print("=" * 65)

    train_img_dir = os.path.join(data_dir, "train", "images")
    train_mask_dir = os.path.join(data_dir, "train", "masks")
    val_img_dir = os.path.join(data_dir, "val", "images")
    val_mask_dir = os.path.join(data_dir, "val", "masks")

    train_ds = CarotidDataset(train_img_dir, train_mask_dir, is_train=True, augment=True, seed=seed)
    val_ds = CarotidDataset(val_img_dir, val_mask_dir, is_train=False, augment=False, seed=seed)

    print(f"Loaded {len(train_ds)} training pairs (with real-time augmentation).")
    print(f"Loaded {len(val_ds)} validation pairs (evaluation only, no augment).")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = UNet(in_channels=1, out_channels=1, base_channels=16).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:       {num_params:,} ({num_params * 4 / (1024 * 1024):.2f} MB)")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_dice": [], "val_iou": [], "lr": []
    }

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * imgs.size(0)

        train_loss = running_train_loss / len(train_ds)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                running_val_loss += loss.item() * imgs.size(0)

                dice_b, iou_b = calculate_metrics(logits, masks)
                val_dice_sum += dice_b * imgs.size(0)
                val_iou_sum += iou_b * imgs.size(0)

        val_loss = running_val_loss / len(val_ds)
        val_dice = val_dice_sum / len(val_ds)
        val_iou = val_iou_sum / len(val_ds)

        epoch_time = time.time() - t_start

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)

        # Checkpoint selection
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0

            os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_loss": val_loss,
                "base_channels": 16,
            }, weights_path)
            tag = " ★ BEST MODEL SAVED"
        else:
            patience_counter += 1
            tag = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] {epoch_time:4.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr:.1e}{tag}")

        # Early stopping check
        if patience_counter >= early_stopping_patience and epoch >= 10:
            print(f"\nEarly stopping triggered after {epoch} epochs (no validation Dice improvement for {early_stopping_patience} epochs).")
            break

    total_training_duration = time.time() - start_total_time
    print(f"\nTraining completed in {total_training_duration/60:.2f} minutes.")
    print(f"Best Validation Dice: {best_val_dice:.4f} achieved at Epoch {best_epoch}.")

    # Save log to CSV
    os.makedirs(os.path.dirname(os.path.abspath(log_csv_path)), exist_ok=True)
    with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                f"{history['train_loss'][i]:.5f}",
                f"{history['val_loss'][i]:.5f}",
                f"{history['val_dice'][i]:.5f}",
                f"{history['val_iou'][i]:.5f}",
                f"{history['lr'][i]:.6f}",
            ])
    print(f"Training log written to: {log_csv_path}")

    # Plot loss and Dice curves
    save_training_curves(history, curves_png_path)

    # Health check rule: "If validation Dice stays below 0.5 after a reasonable attempt, STOP, show me the diagnosis"
    if best_val_dice < 0.5:
        print("\n" + "!" * 65)
        print("CRITICAL WARNING: Best validation Dice score is below 0.50!")
        print(f"Current Best Dice: {best_val_dice:.4f}")
        print("Diagnosis: Model failed to converge on carotid artery lumen boundaries.")
        print("!" * 65)
        sys.exit(1)

    return best_val_dice


def save_training_curves(history: dict, save_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs_range = history["epoch"]

    plt.figure(figsize=(13, 5), facecolor='#121620')

    # Subplot 1: Loss curves
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('#1F2634')
    plt.plot(epochs_range, history["train_loss"], label="Train Loss (BCE+Dice)", color="#24A6FF", linewidth=2.0)
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#FFAE42", linewidth=2.0)
    plt.xlabel("Epoch", color="#AEBCCF", fontsize=11)
    plt.ylabel("Loss", color="#AEBCCF", fontsize=11)
    plt.title("Training & Validation Loss Progression", color="#F2F6FC", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(facecolor="#1F2634", edgecolor="#384458", labelcolor="#F2F6FC")
    ax1.tick_params(colors="#AEBCCF")

    # Subplot 2: Dice & IoU curves
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('#1F2634')
    plt.plot(epochs_range, history["val_dice"], label="Val Dice Score", color="#38D6E1", linewidth=2.2)
    plt.plot(epochs_range, history["val_iou"], label="Val IoU (Jaccard)", color="#34BE70", linewidth=2.0)
    plt.xlabel("Epoch", color="#AEBCCF", fontsize=11)
    plt.ylabel("Metric Score", color="#AEBCCF", fontsize=11)
    plt.title("Validation Segmentation Performance", color="#F2F6FC", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(facecolor="#1F2634", edgecolor="#384458", labelcolor="#F2F6FC")
    ax2.tick_params(colors="#AEBCCF")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Carotid U-Net.")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

