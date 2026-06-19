"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET (BẢN KHÔNG DÙNG SPATIAL ATTENTION)
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder
- Nhập liệu: RGB (3) + Visible mask (1) + Edge mask (1)
- Đầu ra: Amodal mask (1)
- Trạng thái: Đã gỡ bỏ Spatial Attention để khớp với checkpoint no-spatial
===================================================================================
"""

import torch
import torch.nn as nn
import timm
import torchvision.models as models


# ===================================================================================
# KHỐI 1: TỔ HỢP TÍCH CHẬP KÉP (DOUBLE CONVOLUTION BLOCK)
# ===================================================================================
class DoubleConv(nn.Module):
    """
    Khối hai lớp tích chập liên tiếp - thành phần cơ bản của U-Net.
    
    Cấu trúc:
    Conv2d(in→out, 3×3) → BatchNorm → ReLU → Conv2d(out→out, 3×3) → BatchNorm → ReLU
    """
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


# ===================================================================================
# KHỐI 2: KHỐI PHÓNG TỈ LỆ LÊN (UP-SAMPLING BLOCK)
# ===================================================================================
class UpBlock(nn.Module):
    """
    Khối phóng tỉ lệ lên để khôi phục độ phân giải trong quá trình giải mã.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Tích chập chuyển vị (Deconvolution): giảm kênh 1/2, tăng kích thước 2x
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        # Phóng tỉ lệ lên 2 lần
        x_up = self.up(x_decoder)
        # Nối chiều kênh: [skip, upsampled]
        x_concat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_concat)


# ===================================================================================
# KHỐI 3: MÔ HÌNH CHÍNH - AMODAL SWIN-UNET (NO SPATIAL)
# ===================================================================================
class AmodalSwinUNet(nn.Module):
    """
    Mô hình chính để dự đoán hình dạng toàn bộ (Amodal Shape) của vật thể che khuất.
    
    Kiến trúc:
    1. ENCODER: Swin Transformer (đã huấn luyện trước trên ImageNet, nhận 5 kênh)
    2. DECODER: Khôi phục độ phân giải gốc qua 3 lớp up-sampling + skip connections.
    """
    
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 1: ENCODER (Trích xuất đặc trưng)
        # ─────────────────────────────────────────────────────────────────────
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Cải thiệp lớp patch embedding để xử lý 5 kênh (thay vì 3)
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4) 
        
        # Sao chép trọng số pre-trained cho 3 kênh RGB
        with torch.no_grad():
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            # Khởi tạo ngẫu nhiên cho 2 kênh bổ sung (Visible + Edge)
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 2: DECODER U-NET (Khôi phục độ phân giải)
        # ─────────────────────────────────────────────────────────────────────
        self.up1 = UpBlock(768, 384)    # Từ 768 → 384 kênh
        self.up2 = UpBlock(384, 192)    # Từ 384 → 192 kênh
        self.up3 = UpBlock(192, 96)     # Từ 192 → 96 kênh

        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Lớp tích chập cuối cùng: tạo ra dự đoán mask nhị phân (1 kênh)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        """
        Dự đoán mask amodal từ ảnh 5 kênh.
        """
        
        # ──────────────────────────────────────────────────────
        # PHASE 1: ENCODER
        # ──────────────────────────────────────────────────────
        skip_connections = self.encoder(x)

        formatted_skips = []
        for skip in skip_connections:
            formatted_skips.append(skip.permute(0, 3, 1, 2))

        x_bottleneck = formatted_skips[3]

        # ──────────────────────────────────────────────────────
        # PHASE 2: DECODER
        # ──────────────────────────────────────────────────────
        x_decoder = self.up1(x_bottleneck, formatted_skips[2])  
        x_decoder = self.up2(x_decoder, formatted_skips[1])  
        x_decoder = self.up3(x_decoder, formatted_skips[0])  

        x_upsampled = self.up_final(x_decoder)

        # ──────────────────────────────────────────────────────
        # PHASE 3: DỰ ĐOÁN CUỐI CÙNG (Đã bỏ Spatial Attention)
        # ──────────────────────────────────────────────────────
        # Đưa trực tiếp x_upsampled vào lớp chập cuối
        logits = self.final_conv(x_upsampled)
        
        return logits


# ===================================================================================
# PHẦN TEST NHANH 
# ===================================================================================
if __name__ == "__main__":
    model = AmodalSwinUNet()
    
    # Tạo input giả định: 2 bức ảnh, 5 kênh, kích thước 224×224
    dummy_input = torch.randn(2, 5, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"✅ Kiến trúc Swin-UNet 5 kênh (KHÔNG có Spatial Attention) hoạt động OK!")
    print(f"Đầu vào (Ảnh):    {dummy_input.shape}")
    print(f"Đầu ra (Mask):    {output.shape} (Phải là [2, 1, 224, 224])")