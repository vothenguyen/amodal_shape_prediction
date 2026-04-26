"""
===================================================================================
MÔ HÌNH AMODAL THEO BÀI BÁO "Expanding From Active Boundary with Compatible Prior"
===================================================================================
Pipeline gồm 4 module:
1. Model Base (Swin Unet)
2. Active Boundary Estimator
3. Shape Prior Bank
4. Context-Aware Amodal Mask Refiner
===================================================================================
"""

import torch
import torch.nn as nn
import timm

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
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
        if x_skip.size() != x_up.size():
            # Align sizes if necessary
            import torch.nn.functional as F
            x_up = F.interpolate(x_up, size=(x_skip.size(2), x_skip.size(3)), mode='bilinear', align_corners=True)
        x_concat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_concat)

# ===================================================================================
# MODULE 1: MODEL BASE (Swin U-Net)
# Trích xuất đặc trưng và mask nhìn thấy
# ===================================================================================
class ModelBase(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()
        # Backbone ResNet hoặc Swin Transformer
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Sửa đổi cho đầu vào RGB
        self.encoder.patch_embed.proj = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        
        # Decoder đơn giản để tạo ra M^v (Visible Mask)
        self.up1 = UpBlock(768, 384)
        self.up2 = UpBlock(384, 192)
        self.up3 = UpBlock(192, 96)
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mask_v_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        features = self.encoder(x)
        formatted_skips = [f.permute(0, 3, 1, 2) for f in features]
        x_bottleneck = formatted_skips[3]
        
        x_dec = self.up1(x_bottleneck, formatted_skips[2])
        x_dec = self.up2(x_dec, formatted_skips[1])
        x_dec = self.up3(x_dec, formatted_skips[0])
        
        feature_map_F = self.up_final(x_dec)       # Đặc trưng F [B, 64, H, W]
        visible_mask_Mv = self.mask_v_conv(feature_map_F) # Mask M^v [B, 1, H, W]
        return feature_map_F, visible_mask_Mv

# ===================================================================================
# MODULE 2: NON-LOCAL ACTIVE BOUNDARY ESTIMATOR
# Dự đoán ranh giới mở rộng (M^b) dựa trên F và M^v
# ===================================================================================
class ActiveBoundaryEstimator(nn.Module):
    def __init__(self, feature_channels=64):
        super().__init__()
        # Kết hợp F và các thông tin không gian (ở đây mô phỏng bằng chập)
        self.conv_combine = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, F, Mv):
        """
        F: Đặc trưng ảnh (từ Model Base)
        Mv: Lớp biểu diễn visible mask logit
        """
        # Nối đặc trưng và mask
        x = torch.cat([F, Mv], dim=1)
        # Dự đoán ranh giới M^b
        boundary_Mb = self.conv_combine(x)
        return boundary_Mb

# ===================================================================================
# MODULE 3: BOUNDARY-AWARE SHAPE PRIOR BANK
# Truy xuất prior shape (M^p) qua codebook (Mô phỏng dictionary học được)
# ===================================================================================
class ShapePriorBank(nn.Module):
    def __init__(self, num_classes=91, embed_dim=256, codebook_size=512):
        super().__init__()
        self.encoder_Mv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.encoder_Mb = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.query_proj = nn.Linear(32, embed_dim)
        
        # Mô phỏng Shape Prior Codebook (Keys: embeds, Values: decoded shapes)
        self.codebook = nn.Parameter(torch.randn(codebook_size, embed_dim))
        
        # Decoder từ value trả về Shape Prior (M^p) kích thước 224x224
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 14*14*64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 14, 14)),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, Mv, Mb, category_id=None):
        # Trích xuất nhúng ảnh cho M^v và M^b
        feat_v = self.encoder_Mv(Mv).view(Mv.size(0), -1)
        feat_b = self.encoder_Mb(Mb).view(Mb.size(0), -1)
        
        # Tạo Query
        query = self.query_proj(torch.cat([feat_v, feat_b], dim=1)) # [B, embed_dim]
        
        # Dot product attention lên codebook để lấy prior
        attention = torch.matmul(query, self.codebook.T) # [B, codebook_size]
        attention = torch.softmax(attention, dim=-1)
        
        # Vector Value tổng hợp
        retrieved_prior_vector = torch.matmul(attention, self.codebook) # [B, embed_dim]
        
        # Decode thành Shape Prior M^p
        shape_prior_Mp = self.decoder(retrieved_prior_vector)
        return shape_prior_Mp

# ===================================================================================
# MODULE 4: CONTEXT-AWARE AMODAL MASK REFINER
# Kết hợp Feature F, Boundary Mb, và Prior Mp để tạo Amodal Mask (M^a)
# ===================================================================================
class AmodalMaskRefiner(nn.Module):
    def __init__(self, feature_channels=64):
        super().__init__()
        self.context_extractor = nn.Conv2d(feature_channels, 32, kernel_size=3, padding=1)
        self.spatial_attention = SpatialAttention()
        
        # Kết hợp context (32), Mb (1), Mp (1)
        self.combine = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, F, Mb, Mp):
        # Trích xuất bối cảnh từ đặc trưng ảnh
        context = self.context_extractor(F)
        context = self.spatial_attention(context)
        
        # Nối các thông tin lại
        x = torch.cat([context, Mb, Mp], dim=1) # [B, 34, H, W]
        amodal_mask_Ma = self.combine(x)
        return amodal_mask_Ma

# ===================================================================================
# PIPELINE CHÍNH (TỔNG HỢP 4 MODULE)
# ===================================================================================
class AmodalPipelineNguyen(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.model_base = ModelBase()
        self.boundary_estimator = ActiveBoundaryEstimator(feature_channels=64)
        self.shape_prior_bank = ShapePriorBank(num_classes=num_classes)
        self.mask_refiner = AmodalMaskRefiner(feature_channels=64)

    def forward(self, image_x, category_ids=None):
        """
        Input: 
            image_x: [B, 3, 224, 224] (RGB)
        Returns:
            Ma: Amodal Mask (Dự đoán cuối cùng)
            Mv: Visible Mask (Mask phần thấy được)
            Mb: Boundary Mask
            Mp: Shape Prior Mask
        """
        # 1. Trích xuất đặc trưng & M^v
        F, Mv = self.model_base(image_x)
        
        # 2. Dự đoán Boundary M^b
        Mb = self.boundary_estimator(F, torch.sigmoid(Mv))
        
        # 3. Truy xuất Shape Prior M^p
        Mp = self.shape_prior_bank(torch.sigmoid(Mv), torch.sigmoid(Mb), category_ids)
        
        # 4. Tinh chỉnh Amodal Mask M^a
        Ma = self.mask_refiner(F, torch.sigmoid(Mb), torch.sigmoid(Mp))
        
        return Ma, Mv, Mb, Mp

# ===================================================================================
# TEST PIPELINE
# ===================================================================================
if __name__ == "__main__":
    model = AmodalPipelineNguyen()
    test_img = torch.randn(2, 3, 224, 224)
    test_cls = torch.tensor([1, 2])
    
    Ma, Mv, Mb, Mp = model(test_img, test_cls)
    print("✅ Kiểm tra Model Amodal Pipeline Nguyen với 4 module thành công!")
    print(f"Ảnh đầu vào: {test_img.shape}")
    print(f"- Ma (Amodal Mask): {Ma.shape}")
    print(f"- Mv (Visible Mask): {Mv.shape}")
    print(f"- Mb (Boundary Mask): {Mb.shape}")
    print(f"- Mp (Shape Prior): {Mp.shape}")
