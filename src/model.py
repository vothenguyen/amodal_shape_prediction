"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET - Dự đoán hình dạng toàn bộ của vật thể che khuất
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder + Spatial Attention
- Nhập liệu: RGB (3) + Visible mask (1) + Edge mask (1) + Class ID
- Đầu ra: Amodal mask (1)
- Ứng dụng: Hoàn thiện hình dạng của vật thể bị che khuất bằng các vật thể khác
===================================================================================
"""

import torch
import torch.nn as nn
import timm
import torchvision.models as models


# ===================================================================================
# KHỐI 1: CƠ CHẾ CHÚ Ý KHÔNG GIAN (SPATIAL ATTENTION MECHANISM)
# ===================================================================================
class SpatialAttention(nn.Module):
    """
    Cơ chế chú ý không gian - tập trung vào các khu vực quan trọng trong bản đồ đặc trưng.
    
    Hoạt động:
    1. Tính trung bình và max theo chiều kênh
    2. Nối 2 giá trị này lại
    3. Áp dụng tích chập + sigmoid để tạo bản đồ trọng số
    4. Nhân với input để tái cân bằng các kênh
    
    Args:
        kernel_size: Kích thước kernel tích chập (3 hoặc 7, mặc định 7)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kích thước kernel phải là 3 hoặc 7"
        padding = 3 if kernel_size == 7 else 1
        # Lớp tích chập: 2 kênh (avg + max) → 1 kênh trọng số
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # Hàm kích hoạt sigmoid để chuẩn hóa trọng số về [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Tính toán trọng số chú ý và nhân với input.
        
        Args:
            x: Đầu vào [Batch, Channels, Height, Width]
        
        Returns:
            Đầu ra được tái cân bằng theo trọng số chú ý
        """
        # Tính giá trị trung bình theo các kênh
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Tính giá trị cực đại theo các kênh
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Nối hai giá trị để có 2 kênh
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Tạo bản đồ trọng số chú ý
        scale = self.sigmoid(self.conv1(x_cat))
        # Tái cân bằng: nhân từng phần tử với trọng số tương ứng
        return x * scale


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
       - Xử lý 5 kênh: RGB (3) + Visible mask (1) + Edge mask (1)
       - Trích xuất đặc trưng phân cấp
    
    2. EMBEDDING NHÃN: Chuyển đổi class ID thành vector nhúng
       - Hỗ trợ 91 loại vật thể COCO
       - Nhúng vào bottleneck của U-Net
    
    3. DECODER: Khôi phục độ phân giải gốc
       - Sử dụng skip connections từ encoder
       - Gồm 3 lớp up-sampling
    
    4. SPATIAL ATTENTION: Tập trung vào vùng quan trọng
       - Cơ chế chú ý không gian
       - Tăng độ chính xác dự đoán
    
    Args:
        model_name: Tên mô hình encoder từ timm (mặc định: swin_tiny_patch4_window7_224)
        pretrained: Có dùng trọng số pre-trained không (mặc định: True)
        num_classes: Số loại vật thể (mặc định: 91 cho COCO)
    """
    
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, num_classes=91):
        super().__init__()

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 1: ENCODER (Trích xuất đặc trưng)
        # ─────────────────────────────────────────────────────────────────────
        # Tạo mô hình Swin Transformer encoder đã huấn luyện trước
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Cải thiệp lớp patch embedding để xử lý 5 kênh (thay vì 3)
        # Lưu trọng số gốc cho 3 kênh RGB
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        # Tạo lớp tích chập mới cho 5 kênh
        self.encoder.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4) 
        
        # Sao chép trọng số pre-trained cho 3 kênh RGB
        with torch.no_grad():
            # Các trọng số cho R, G, B từ mô hình gốc
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            # Khởi tạo ngẫu nhiên cho 2 kênh bổ sung (Visible + Edge)
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 2: EMBEDDING NHÃN (Class ID → Vector nhúng)
        # ─────────────────────────────────────────────────────────────────────
        # Chuyển đổi ID loại vật thể thành vector nhúng
        # Ví dụ: class_id=3 (xe hơi) → vector 768 chiều
        self.category_emb = nn.Embedding(num_classes, 768)

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 3: CƠ CHẾ CHÚ Ý (Spatial Attention)
        # ─────────────────────────────────────────────────────────────────────
        # Áp dụng tại cửa ra trước lớp tích chập cuối
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # ─────────────────────────────────────────────────────────────────────
        # PHẦN 4: DECODER U-NET (Khôi phục độ phân giải)
        # ─────────────────────────────────────────────────────────────────────
        # 3 lớp up-sampling + skip connections
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


    def forward(self, x, class_ids):
        """
        Dự đoán mask amodal từ ảnh 5 kênh và ID loại vật thể.
        
        Quy trình:
        1. Encoder: Trích xuất đặc trưng phân cấp
        2. Nhúng nhãn vào bottleneck
        3. Decoder: Khôi phục độ phân giải
        4. Spatial Attention: Tập trung vào vùng quan trọng
        5. Final Conv: Tạo ra dự đoán cuối cùng
        
        Args:
            x: Ảnh 5 kênh [Batch, 5, 224, 224]
               - Kênh 0-2: RGB
               - Kênh 3: Visible mask
               - Kênh 4: Edge mask
            class_ids: ID loại vật thể [Batch]
        
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
        # PHASE 2: NHÚNG NHÃN VÀO BOTTLENECK
        # ──────────────────────────────────────────────────────
        # Chuyển đổi class ID → vector nhúng
        c_emb = self.category_emb(class_ids)  # [Batch, 768]
        # Kéo giãn để match với hình dạng bottleneck
        c_emb = c_emb.unsqueeze(-1).unsqueeze(-1)  # [Batch, 768, 1, 1]
        # Cộng nhúng nhãn vào bottleneck để "gợi ý" cho mô hình
        x_bottleneck = x_bottleneck + c_emb  # Broadcasting cộng

        # ──────────────────────────────────────────────────────
        # PHASE 3: DECODER - Khôi phục độ phân giải
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
        # PHASE 4: ÁP DỤNG SPATIAL ATTENTION & DƯỚI ĐỨC
        # ──────────────────────────────────────────────────────
        # Áp dụng cơ chế chú ý không gian để tập trung vào vùng quan trọng
        attended_features = self.spatial_attention(x_upsampled)
        # Tạo ra dự đoán mask cuối cùng (logit chưa qua sigmoid)
        logits = self.final_conv(attended_features)
        
        return logits


# ===================================================================================
# PHẦN TEST NHANH - Kiểm tra kiến trúc
# ===================================================================================
if __name__ == "__main__":
    # Tạo mô hình
    model = AmodalSwinUNet()
    
    # Tạo input giả định: 2 bức ảnh, 5 kênh, kích thước 224×224
    dummy_input = torch.randn(2, 5, 224, 224)
    # Tạo class IDs giả định: ảnh 1 là loại 3, ảnh 2 là loại 1
    dummy_class = torch.tensor([3, 1]) 
    
    # Chạy qua mô hình
    with torch.no_grad():
        output = model(dummy_input, dummy_class)
        
    # In kết quả
    print(f"✅ Kiến trúc Swin-UNet 5 kênh + Nhúng nhãn + Spatial Attention hoạt động OK!")
    print(f"Đầu vào (Ảnh):    {dummy_input.shape}")
    print(f"Đầu vào (Nhãn):   {dummy_class.shape}")
    print(f"Đầu ra (Mask):    {output.shape} (Phải là [2, 1, 224, 224])")