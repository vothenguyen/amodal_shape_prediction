"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET (ROW 7 COCOA — PROPOSED: SG-GBA)
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder 
           + Semantic-Guided Gated Bottleneck Attention (SG-GBA)
- Nhập liệu: RGB (3) + Visible mask (1) + Edge mask (1) + Class ID (1)
- Đầu ra: Amodal mask logits (1)
- Category Embedding: 91 lớp COCO
- Bottleneck Attention: SGGatedBottleneck (FiLM điều biến ngữ nghĩa + Chú ý không gian 7x7 + Residual Gate)
===================================================================================
"""

import torch
import torch.nn as nn
import timm


class SGGatedBottleneck(nn.Module):
    """
    Semantic-Guided Gated Bottleneck Attention (SG-GBA - Row 7).
    Kết hợp điều biến đặc trưng theo ngữ nghĩa (FiLM) và chú ý không gian toàn cục
    tại độ phân giải 7x7 với cổng nối tắt học được (Learnable Residual Gate).
    """
    def __init__(self, channels=768, reduction=4, spatial_kernel=3):
        super().__init__()
        hidden_dim = channels // reduction  # 768 // 4 = 192
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
        # Khởi tạo alpha = 0.01 để đảm bảo cả alpha, film_mlp và spatial_conv
        # đều nhận được gradient khác 0 ngay từ batch đầu tiên
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.last_attn_map = None

        # Zero-init lớp Linear cuối của FiLM để tại step đầu f_mod == x_bottleneck
        nn.init.zeros_(self.film_mlp[2].weight)
        nn.init.zeros_(self.film_mlp[2].bias)

    def forward(self, x_bottleneck, c_emb):
        # x_bottleneck: [B, 768, 7, 7], c_emb: [B, 768]
        film_params = self.film_mlp(c_emb).unsqueeze(-1).unsqueeze(-1)  # [B, 1536, 1, 1]
        gamma, beta = torch.chunk(film_params, 2, dim=1)               # Mỗi tensor [B, 768, 1, 1]

        # 1. Điều biến kênh theo ngữ nghĩa (FiLM)
        f_mod = (1.0 + torch.tanh(gamma)) * x_bottleneck + beta

        # 2. Chú ý không gian toàn cục tại Bottleneck (7x7)
        avg_out = torch.mean(f_mod, dim=1, keepdim=True)               # [B, 1, 7, 7]
        max_out, _ = torch.max(f_mod, dim=1, keepdim=True)             # [B, 1, 7, 7]
        attn_map = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )  # [B, 1, 7, 7]

        # Lưu lại attention map phục vụ trực quan hóa Heatmap trên Gradio UI (app.py)
        self.last_attn_map = attn_map.detach()

        # 3. Kết nối tắt có cổng học được (Learnable Residual Gate)
        out = x_bottleneck + self.alpha * (f_mod * attn_map)
        return out


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
    Row 7: Swin-UNet với Semantic-Guided Gated Bottleneck Attention (SG-GBA) trên COCOA:
    5 kênh + Category Embedding (91 lớp) + SGGatedBottleneck.
    """
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, num_classes=91):
        super().__init__()

        # Encoder: Swin Transformer 5 kênh
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4)

        with torch.no_grad():
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        # Category Embedding: 91 lớp cho COCO
        self.category_emb = nn.Embedding(num_classes, 768)

        # Module Row 7: Semantic-Guided Gated Bottleneck Attention
        self.sg_bottleneck = SGGatedBottleneck(channels=768, reduction=4, spatial_kernel=3)

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

    def forward(self, x, class_ids):
        """
        Dự đoán mask amodal từ ảnh 5 kênh và class_ids.
        x: [B, 5, 224, 224]
        class_ids: [B]
        """
        skip_connections = self.encoder(x)
        formatted_skips = [s.permute(0, 3, 1, 2) for s in skip_connections]

        # Lấy đặc trưng sâu nhất ở bottleneck [B, 768, 7, 7]
        x_bottleneck = formatted_skips[3]

        # Vector nhúng nhãn ngữ nghĩa [B, 768]
        c_emb = self.category_emb(class_ids)

        # Điều biến và gating SG-GBA tại bottleneck
        x_bottleneck = self.sg_bottleneck(x_bottleneck, c_emb)

        # Decoder U-Net qua 3 cấp phóng tỉ lệ
        x_decoder = self.up1(x_bottleneck, formatted_skips[2])
        x_decoder = self.up2(x_decoder, formatted_skips[1])
        x_decoder = self.up3(x_decoder, formatted_skips[0])

        x_upsampled = self.up_final(x_decoder)
        logits = self.final_conv(x_upsampled)
        return logits


if __name__ == "__main__":
    EXPECTED_PARAMS = 34_867_566
    model = AmodalSwinUNet(pretrained=False, num_classes=91)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Row 7 COCOA Parameters: {total_params:,} (Kỳ vọng: {EXPECTED_PARAMS:,})")
    assert total_params == EXPECTED_PARAMS, f"Lệch tham số: {total_params} != {EXPECTED_PARAMS}"

    dummy = torch.randn(2, 5, 224, 224)
    cls = torch.tensor([1, 90], dtype=torch.long)
    out = model(dummy, cls)
    print("Row 7 COCOA Model Output shape:", out.shape)
    assert out.shape == (2, 1, 224, 224), "Output shape không hợp lệ!"
    print("✅ Kiểm tra kiến trúc Row 7 COCOA thành công!")
