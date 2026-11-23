"""U-Net Architecture for Carotid Artery Lumen Segmentation.

Owner: Srujan (Module 1)
Designed for real-time ultrasound lumen segmentation.
Lightweight configuration (under 1M parameters) optimized for CPU/GPU efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D 3x3 -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling with Bilinear Interpolation or ConvTranspose2d then DoubleConv with Skip Concatenation"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if spatial dimensions have odd differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Compact 2D U-Net for carotid artery lumen segmentation.

    Args:
        in_channels: Input channels (1 for grayscale B-mode ultrasound).
        out_channels: Output channels (1 for binary lumen logits).
        base_channels: Base feature channels (default 16 for <1M parameter footprint).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_channels)          # 16
        self.down1 = Down(base_channels, base_channels * 2)             # 32
        self.down2 = Down(base_channels * 2, base_channels * 4)         # 64
        self.bottleneck = Down(base_channels * 4, base_channels * 8)    # 128

        self.up1 = Up(base_channels * 8 + base_channels * 4, base_channels * 4, bilinear=True) # 128+64 -> 64
        self.up2 = Up(base_channels * 4 + base_channels * 2, base_channels * 2, bilinear=True) # 64+32 -> 32
        self.up3 = Up(base_channels * 2 + base_channels, base_channels, bilinear=True)         # 32+16 -> 16
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)       # (B, 16, 256, 256)
        x2 = self.down1(x1)        # (B, 32, 128, 128)
        x3 = self.down2(x2)        # (B, 64, 64, 64)
        x4 = self.bottleneck(x3)   # (B, 128, 32, 32)

        x = self.up1(x4, x3)       # (B, 64, 64, 64)
        x = self.up2(x, x2)        # (B, 32, 128, 128)
        x = self.up3(x, x1)        # (B, 16, 256, 256)
        logits = self.out_conv(x)  # (B, 1, 256, 256)
        return logits


if __name__ == "__main__":
    net = UNet(in_channels=1, out_channels=1, base_channels=16)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    dummy_x = torch.randn(2, 1, 256, 256)
    out = net(dummy_x)
    print(f"U-Net Architecture instantiated successfully.")
    print(f"Input shape:        {dummy_x.shape}")
    print(f"Output shape:       {out.shape}")
    print(f"Total parameters:   {total_params:,} ({total_params*4/(1024*1024):.2f} MB)")
    assert total_params < 1_000_000, f"Parameter count {total_params} exceeds 1M limit!"
