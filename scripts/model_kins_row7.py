"""
===================================================================================
MÔ HÌNH ROW 7: SG-GBA SWIN-UNET CHO KINS (20 CLASSES)
===================================================================================
Kiến trúc: Swin-UNet 5 kênh + Semantic-Guided Gated Bottleneck Attention (SG-GBA)
- Đầu vào: 5 kênh (RGB 3ch + Visible 1ch + Edge 1ch) + Category ID
- Category Embedding: 20 lớp KINS
- Bottleneck Attention: SGGatedBottleneck (FiLM điều biến + Chú ý không gian 7x7 + Residual Gate)
- Decoder: U-Net Decoder 3 UpBlocks, tắt Spatial Attention ở cuối Decoder
- Số lượng tham số chuẩn: EXPECTED_PARAMS = 34,813,038
===================================================================================
"""

import torch
import torch.nn as nn
import timm


class SGGatedBottleneck(nn.Module):
    """
    Semantic-Guided Gated Bottleneck Attention (SG-GBA).
    1. FiLM điều biến kênh đặc trưng qua vector ngữ nghĩa category_emb
    2. Chú ý không gian Spatial Attention qua Conv2d 3x3
    3. Gated Residual connection với trọng số học được alpha (khởi tạo 0.01)
    """
    def __init__(self, channels=768, reduction=4, spatial_kernel=3):
        super().__init__()
        hidden_dim = channels // reduction
        self.film_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2, bias=True),
        )
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.last_attn_map = None

        # Khởi tạo identity cho FiLM (gamma=0, beta=0)
        nn.init.zeros_(self.film_mlp[2].weight)
        nn.init.zeros_(self.film_mlp[2].bias)

    def forward(self, x_bottleneck, c_emb):
        film_params = self.film_mlp(c_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        f_mod = (1.0 + torch.tanh(gamma)) * x_bottleneck + beta
        avg_out = torch.mean(f_mod, dim=1, keepdim=True)
        max_out, _ = torch.max(f_mod, dim=1, keepdim=True)
        attn_map = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )
        self.last_attn_map = attn_map.detach()
        out = x_bottleneck + self.alpha * (f_mod * attn_map)
        return out


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        x_up = self.up(x_decoder)
        x_concat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_concat)


class AmodalSwinUNetRow7KINS(nn.Module):
    """
    Amodal Swin-UNet Row 7 cho tập KINS (20 classes).
    Tổng tham số chuẩn: 34,813,038
    """
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, num_classes=20):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4)

        with torch.no_grad():
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        self.category_emb = nn.Embedding(num_classes, 768)
        # Khối SG-GBA với Conv2d 3x3 (444,115 params trong bottleneck SG-GBA)
        self.sg_bottleneck = SGGatedBottleneck(channels=768, reduction=4, spatial_kernel=3)

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

    def forward(self, x, class_ids):
        skip_connections = self.encoder(x)
        formatted_skips = [s.permute(0, 3, 1, 2) for s in skip_connections]
        x_bottleneck = formatted_skips[3]
        c_emb = self.category_emb(class_ids)
        x_bottleneck = self.sg_bottleneck(x_bottleneck, c_emb)
        x_decoder = self.up1(x_bottleneck, formatted_skips[2])
        x_decoder = self.up2(x_decoder, formatted_skips[1])
        x_decoder = self.up3(x_decoder, formatted_skips[0])
        x_upsampled = self.up_final(x_decoder)
        logits = self.final_conv(x_upsampled)
        return logits


if __name__ == "__main__":
    NUM_CLASSES = 20
    EXPECTED_PARAMS = 34_813_038
    model = AmodalSwinUNetRow7KINS(pretrained=False, num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Row 7 KINS (num_classes={NUM_CLASSES}) Params: {total_params:,} (Kỳ vọng: {EXPECTED_PARAMS:,})")
    assert total_params == EXPECTED_PARAMS, f"Lệch tham số: {total_params} != {EXPECTED_PARAMS}"
    print("✅ Kiểm tra số lượng tham số Row 7 KINS thành công 100%!")
