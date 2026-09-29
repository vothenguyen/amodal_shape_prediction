"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET (ROW 3 D2SA — 5 KÊNH, KHÔNG EMBEDDING, KHÔNG SPATIAL)
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder
- Nhập liệu: RGB (3) + Visible mask (1) + Edge mask (1)
- Đầu ra: Amodal mask (1)
- Loss: OcclusionAwareLoss (5x trọng số vùng che)
===================================================================================
"""

import torch
import torch.nn as nn
import timm


class DoubleConv(nn.Module):
    """Khối hai lớp tích chập liên tiếp trong U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    """Khối phóng tỉ lệ lên kết hợp skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        x_up = self.up(x_decoder)
        x_concat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_concat)


class AmodalSwinUNet(nn.Module):
    """
    Row 3: Swin-UNet 5 kênh (RGB + Vis Mask + Edge Mask), không embedding, không spatial attention.
    """
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()

        # Encoder: Swin Transformer 5 kênh
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4)

        with torch.no_grad():
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        # Decoder U-Net
        self.up1 = UpBlock(768, 384)
        self.up2 = UpBlock(384, 192)
        self.up3 = UpBlock(192, 96)

        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        """
        Dự đoán mask amodal từ ảnh 5 kênh.
        x: [B, 5, 224, 224]
        """
        skip_connections = self.encoder(x)
        formatted_skips = [s.permute(0, 3, 1, 2) for s in skip_connections]

        x_bottleneck = formatted_skips[3]

        x_decoder = self.up1(x_bottleneck, formatted_skips[2])
        x_decoder = self.up2(x_decoder, formatted_skips[1])
        x_decoder = self.up3(x_decoder, formatted_skips[0])

        x_upsampled = self.up_final(x_decoder)
        logits = self.final_conv(x_upsampled)
        return logits


if __name__ == "__main__":
    model = AmodalSwinUNet(pretrained=False)
    dummy = torch.randn(2, 5, 224, 224)
    out = model(dummy)
    print("Row 3 D2SA Model Output shape:", out.shape)
