"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET - Dự đoán hình dạng toàn bộ của vật thể bị che khuất
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder
- Nhập liệu: RGB (3) + Visible mask (1) + Class ID
- Đầu ra: Amodal mask (1 kênh)
- Ứng dụng: Hoàn thiện hình dạng của vật thể bị che khuất bằng các vật thể khác
===================================================================================
"""

import torch
import torch.nn as nn
import timm
import torchvision.models as models

# ===================================================================================
# KHỐI 2: TỔ HỢP TÍCH CHẬP KÉP (DOUBLE CONVOLUTION BLOCK)
# ===================================================================================
class DoubleConv(nn.Module):
    """
    Khối hai lớp tích chập liên tiếp - thành phần cơ bản của U-Net.
    
    Cấu trúc:
    Conv2d(in→out, 3×3) → BatchNorm → ReLU → Conv2d(out→out, 3×3) → BatchNorm → ReLU
    
    Args:
        in_channels: Số kênh đầu vào
        out_channels: Số kênh đầu ra
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Lớp tích chập thứ nhất: mở rộng từ in_channels → out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Lớp tích chập thứ hai: giữ out_channels → out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Đi qua toàn bộ khối tích chập kép."""
        return self.double_conv(x)


# ===================================================================================
# KHỐI 3: KHỐI PHÓ NG TỈ LỆ LÊN (UP-SAMPLING BLOCK)
# ===================================================================================
class UpBlock(nn.Module):
    """
    Khối phóng tỉ lệ lên để khôi phục độ phân giải trong quá trình giải mã.
    
    Hoạt động:
    1. Sử dụng ConvTranspose2d để tăng 2 lần kích thước không gian
    2. Nối (concatenate) với skip connection từ encoder
    3. Áp dụng DoubleConv để xử lý đầu ra kết hợp
    
    Args:
        in_channels: Số kênh của feature map giải mã
        out_channels: Số kênh đầu ra
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Tích chập chuyển vị (Deconvolution): giảm kênh 1/2, tăng kích thước 2x
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Xử lý sau khi nối với skip connection
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        """
        Phóng tỉ lệ lên và nối với skip connection.
        
        Args:
            x_decoder: Feature map từ lớp giải mã trước đó
            x_skip: Skip connection từ encoder tương ứng
        
        Returns:
            Feature map sau xử lý
        """
        # Phóng tỉ lệ lên 2 lần
        x_up = self.up(x_decoder)
        # Nối chiều kênh: [skip, upsampled]
        x_concat = torch.cat([x_skip, x_up], dim=1)
        # Xử lý tích chập trên kết hợp
        return self.conv(x_concat)


# ===================================================================================
# KHỐI 4: MÔ HÌNH CHÍNH - AMODAL SWIN-UNET PRO MAX
# ===================================================================================
class AmodalSwinUNet(nn.Module):
    """
    Mô hình chính để dự đoán hình dạng toàn bộ (Amodal Shape) của vật thể che khuất.
    
    Kiến trúc:
    1. ENCODER: Swin Transformer (đã huấn luyện trước trên ImageNet)
       - Xử lý 4 kênh đầu vào: RGB (3) + Visible mask (1)
       - Trích xuất đặc trưng phân cấp
    
    2. DECODER: Khôi phục độ phân giải gốc
       - Sử dụng skip connections từ encoder
       - Gồm 3 lớp up-sampling
    
    Args:
        model_name: Tên mô hình encoder từ timm (mặc định: swin_tiny_patch4_window7_224)
        pretrained: Có dùng trọng số pre-trained không (mặc định: True)
    """
    
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 1: ENCODER (Trích xuất đặc trưng)
        # ─────────────────────────────────────────────────────────────────────
        # Tạo mô hình Swin Transformer encoder đã huấn luyện trước
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Cải thiệp lớp patch embedding để xử lý 4 kênh (thay vì 3)
        # Lưu trọng số gốc cho 3 kênh RGB
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        # Tạo lớp tích chập mới cho 4 kênh
        self.encoder.patch_embed.proj = nn.Conv2d(4, 96, kernel_size=4, stride=4) 
        
        # Sao chép trọng số pre-trained cho 3 kênh RGB
        with torch.no_grad():
            # Các trọng số cho R, G, B từ mô hình gốc
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            # Khởi tạo ngẫu nhiên cho kênh bổ sung (Visible)
            self.encoder.patch_embed.proj.weight[:, 3, :, :] = 0

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 2: DECODER U-NET (Khôi phục độ phân giải)
        # ─────────────────────────────────────────────────────────────────────
        # 3 lớp up-sampling kết hợp skip connections
        self.up1 = UpBlock(768, 384)    # Từ 768 → 384 kênh
        self.up2 = UpBlock(384, 192)    # Từ 384 → 192 kênh
        self.up3 = UpBlock(192, 96)     # Từ 192 → 96 kênh

        # Lớp cuối cùng để phóng tỉ lệ từ 224→224 (Upsampling ×4)
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
        
        Quy trình:
        1. Encoder: Trích xuất đặc trưng phân cấp
        2. Decoder: Khôi phục độ phân giải
        3. Final Conv: Tạo ra dự đoán cuối cùng
        
        Args:
            x: Ảnh 4 kênh [Batch, 4, 224, 224]
               - Kênh 0-2: RGB
               - Kênh 3: Visible mask
        
        Returns:
            Logit mask amodal [Batch, 1, 224, 224]
        """
        
        # ──────────────────────────────────────────────────────
        # PHASE 1: ENCODER - Trích xuất đặc trưng phân cấp
        # ──────────────────────────────────────────────────────
        # Encoder trả về 4 level đặc trưng (từ sâu đến nông)
        skip_connections = self.encoder(x)

        # Định dạng lại skip connections từ [B, H, W, C] → [B, C, H, W]
        formatted_skips = []
        for skip in skip_connections:
            formatted_skips.append(skip.permute(0, 3, 1, 2))

        # Rút xuống đặc trưng ở bottleneck (sâu nhất, độ phân giải thấp nhất)
        x_bottleneck = formatted_skips[3]

        # ──────────────────────────────────────────────────────
        # PHASE 2: DECODER - Khôi phục độ phân giải
        # ──────────────────────────────────────────────────────
        # Lớp 1: 768 → 384 kênh (kích thước ×2)
        x_decoder = self.up1(x_bottleneck, formatted_skips[2])  
        # Lớp 2: 384 → 192 kênh (kích thước ×2)
        x_decoder = self.up2(x_decoder, formatted_skips[1])  
        # Lớp 3: 192 → 96 kênh (kích thước ×2)
        x_decoder = self.up3(x_decoder, formatted_skips[0])  

        # Phóng tỉ lệ cuối cùng từ 56×56 → 224×224
        x_upsampled = self.up_final(x_decoder)

        # ──────────────────────────────────────────────────────
        # PHASE 3: DỰ ĐOÁN ĐẦU RA CUỐI CÙNG
        # ──────────────────────────────────────────────────────
        # Tạo ra dự đoán mask cuối cùng (logit chưa qua sigmoid)
        logits = self.final_conv(x_upsampled) # Truyền thẳng x_upsampled vào lớp cuối
        
        return logits


# ===================================================================================
# PHẦN TEST NHANH - Kiểm tra kiến trúc
# ===================================================================================
if __name__ == "__main__":
    # Tạo mô hình
    model = AmodalSwinUNet()
    
    # Tạo input giả định: 2 bức ảnh, 4 kênh, kích thước 224×224
    dummy_input = torch.randn(2, 4, 224, 224)
    
    # Chạy qua mô hình
    with torch.no_grad():
        output = model(dummy_input)
        
    # In kết quả
    print(f"✅ Kiến trúc Swin-UNet 4 kênh (không nhúng nhãn) hoạt động OK!")
    print(f"Đầu vào (Ảnh):    {dummy_input.shape}")
    print(f"Đầu ra (Mask):    {output.shape} (Phải là [2, 1, 224, 224])")