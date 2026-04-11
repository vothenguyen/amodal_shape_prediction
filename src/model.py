import torch
import torch.nn as nn
import timm
import torchvision.models as models

# --- CÁC KHỐI XÂY DỰNG CHỨC NĂNG (BUILDING BLOCKS) ---

# ==========================================
# 1. MẮT THẦN KHÔNG GIAN (SPATIAL ATTENTION)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv1(x_cat))
        return x * scale

# ==========================================
# 2. Khối Chập Kép (Double Convolution)
# ==========================================
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

# ==========================================
# 3. Khối Phóng To (Up Block)
# ==========================================
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        x_up = self.up(x_decoder)
        x_concat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_concat)


# ==========================================
# --- MÔ HÌNH CHÍNH (AMODAL SWIN-UNET PRO MAX) ---
# ==========================================
class AmodalSwinUNet(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, num_classes=91):
        super().__init__()

        # 1. ENCODER (5 KÊNH)
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4) 
        
        with torch.no_grad():
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        # 2. BỘ GIẢI MÃ NHÃN (CATEGORY EMBEDDING) - Nhúng vào Bottleneck
        self.category_emb = nn.Embedding(num_classes, 768)

        # 3. MẮT THẦN (SPATIAL ATTENTION) - Đặt ở cửa ra
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # 4. DECODER U-NET
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

    # Đón 2 đầu vào: x (Ảnh 5 Kênh) và class_ids (Nhãn vật thể)
    def forward(self, x, class_ids):
        # --- PHASE 1: ENCODER ---
        skip_connections = self.encoder(x)

        formatted_skips = []
        for skip in skip_connections:
            formatted_skips.append(skip.permute(0, 3, 1, 2))

        # Rút trích Bottleneck (Lõi cô đặc nhất)
        x_bottleneck = formatted_skips[3]

        # --- BƠM NHÃN VÀO LÕI ---
        c_emb = self.category_emb(class_ids) # [Batch, 768]
        c_emb = c_emb.unsqueeze(-1).unsqueeze(-1) # Kéo giãn thành [Batch, 768, 1, 1]
        x_bottleneck = x_bottleneck + c_emb # Hợp thể!

        # --- PHASE 2: DECODER ---
        x_decoder = self.up1(x_bottleneck, formatted_skips[2])  
        x_decoder = self.up2(x_decoder, formatted_skips[1])  
        x_decoder = self.up3(x_decoder, formatted_skips[0])  

        x_upsampled = self.up_final(x_decoder)

        # --- PHASE 3: RA KẾT QUẢ ---
        attended_features = self.spatial_attention(x_upsampled) # Bật mắt thần
        logits = self.final_conv(attended_features) # Chốt hạ vẽ nét
        
        return logits

# --- PHẦN TEST NHANH ---
if __name__ == "__main__":
    model = AmodalSwinUNet()
    
    # Giả lập input: 2 bức ảnh, 5 Kênh, 224x224
    dummy_input = torch.randn(2, 5, 224, 224)
    # Giả lập nhãn: Bức 1 là class 3, Bức 2 là class 1
    dummy_class = torch.tensor([3, 1]) 
    
    with torch.no_grad():
        output = model(dummy_input, dummy_class)
        
    print(f"✅ Kiến trúc U-Net 5 Kênh + Bơm Nhãn + Mắt Thần đã OK!")
    print(f"Đầu vào Ảnh: {dummy_input.shape}")
    print(f"Đầu vào Nhãn: {dummy_class.shape}")
    print(f"Đầu ra (Mask): {output.shape} (Phải là 2x1x224x224)")