"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET (ROW 5 D2SA — 5 KÊNH, CÓ SPATIAL ATTENTION, KHÔNG EMBEDDING)
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder + Spatial Attention
- Nhập liệu: RGB (3) + Visible mask (1) + Edge mask (1)
- Đầu ra: Amodal mask (1)
- Có Spatial Attention: tập trung vào các vùng không gian quan trọng trước lớp conv cuối
- Không có Category Embedding
===================================================================================
"""

import torch
import torch.nn as nn
import timm


class SpatialAttention(nn.Module):
    """
    Cơ chế chú ý không gian - tập trung vào các khu vực quan trọng trong feature map.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kích thước kernel phải là 3 hoặc 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv1(x_cat))
        return x * scale


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
    Row 5: Swin-UNet 5 kênh, có Spatial Attention, KHÔNG Category Embedding.
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

        # Cơ chế chú ý không gian
        self.spatial_attention = SpatialAttention(kernel_size=7)

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

        # Áp dụng Spatial Attention
        x_spatial = self.spatial_attention(x_upsampled)

        logits = self.final_conv(x_spatial)
        return logits


if __name__ == "__main__":
    model = AmodalSwinUNet(pretrained=False)
    dummy = torch.randn(2, 5, 224, 224)
    out = model(dummy)
    print("Row 5 D2SA Model Output shape:", out.shape)
