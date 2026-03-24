import torch
import torch.nn as nn
import timm

# --- CÁC KHỐI XÂY DỰNG CHỨC NĂNG (BUILDING BLOCKS) ---


# 1. Khối Chập Kép (Double Convolution) - Linh hồn của U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Lớp chập 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # Chuẩn hóa để train nhanh hơn
            nn.ReLU(inplace=True),  # Hàm kích hoạt ReLU
            # Lớp chập 2 (Làm mượt nét vẽ)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# 2. Khối Phóng To (Up-sampling) có kẹp Skip Connections
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Dùng ConvTranspose2d để phóng to x2, giữ lại thông tin không gian
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        # Khối Chập Kép để xử lý sau khi kẹp (Concat)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        # Bước 1: Phóng to bức ảnh từ Decoder
        x_up = self.up(x_decoder)

        # Bước 2: KẸP (CONCAT) - Đây là bí thuật U-Net!
        # Kẹp bức ảnh mờ mờ vừa phóng to (x_up) với
        # bức ảnh chi tiết sắc nét từ Encoder (x_skip) theo chiều kênh màu
        x_concat = torch.cat([x_skip, x_up], dim=1)

        # Bước 3: Đưa qua khối chập kép để vẽ lại nét mượt mà
        return self.conv(x_concat)


# --- MÔ HÌNH CHÍNH (MAIN MODEL) ---


class AmodalSwinUNet(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()

        # 1. ENCODER: Swin Transformer (Vẫn giữ nguyên như bản cũ)
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, features_only=True
        )

        # Độ lại lớp Patch Embedding đầu vào (Lên 5 kênh!)
        pretrained_patch_embed = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(
            5, 96, kernel_size=4, stride=4
        )  # SỬA SỐ 4 THÀNH 5 Ở ĐÂY
        with torch.no_grad():
            # Copy tạng não 3 kênh màu cũ
            self.encoder.patch_embed.proj.weight[:, :3, :, :] = pretrained_patch_embed
            # Khởi tạo Kênh 4 (Visible) và Kênh 5 (Edge) bằng số 0 để nó từ từ học
            self.encoder.patch_embed.proj.weight[:, 3:, :, :] = 0

        # Các kênh đầu ra của Swin Tiny: [96, 192, 384, 768] (Bản đồ đặc trưng từ to đến bé)
        self.encoder_channels = self.encoder.feature_info.channels()

        # 2. DECODER: U-Net Xịn (Nâng cấp nằm ở đây!)

        # Khối 1: Phóng to từ 7x7 (kênh 768) lên 14x14, kẹp với 14x14 (kênh 384)
        self.up1 = UpBlock(768, 384)
        # Khối 2: Phóng to từ 14x14 (kênh 384) lên 28x28, kẹp với 28x28 (kênh 192)
        self.up2 = UpBlock(384, 192)
        # Khối 3: Phóng to từ 28x28 (kênh 192) lên 56x56, kẹp với 56x56 (kênh 96)
        self.up3 = UpBlock(192, 96)

        # Khối 4: Phóng to cuối cùng từ 56x56 lên 224x224 (Kéo giãn 4 lần bằng Bilinear)
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3. LỚP RA CUỐI CÙNG (Segmentation Head): Vẽ ra Mask nhị phân
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # --- PHASE 1: ENCODER (Đi xuống - Rút trích đặc trưng) ---
        skip_connections = self.encoder(x)

        # 🚨 FIX LỖI Ở ĐÂY: Nắn lại xương cho Swin Transformer!
        # Swin nhả ra [Batch, H, W, Channels], Conv2d của U-Net lại cần [Batch, Channels, H, W]
        # Mình dùng permute(0, 3, 1, 2) để đảo chiều Channels lên đúng vị trí
        formatted_skips = []
        for skip in skip_connections:
            formatted_skips.append(skip.permute(0, 3, 1, 2))

        # formatted_skips[0] -> 56x56 (kênh 96)
        # formatted_skips[1] -> 28x28 (kênh 192)
        # formatted_skips[2] -> 14x14 (kênh 384)
        # formatted_skips[3] -> 7x7 (kênh 768)

        # Bức ảnh bé tí, mờ nhất nằm ở cuối cùng
        x_bottleneck = formatted_skips[3]

        # --- PHASE 2: DECODER (Đi lên - Phóng to và giữ chi tiết) ---
        # Phóng to và kẹp với các "mật lệnh" chi tiết tương ứng
        x_decoder = self.up1(x_bottleneck, formatted_skips[2])  # Lên 14x14
        x_decoder = self.up2(x_decoder, formatted_skips[1])  # Lên 28x28
        x_decoder = self.up3(x_decoder, formatted_skips[0])  # Lên 56x56

        # Phóng to cuối cùng về kích thước ảnh gốc 224x224
        x_upsampled = self.up_final(x_decoder)

        # --- PHASE 3: RA KẾT QUẢ ---
        return self.segmentation_head(x_upsampled)


if __name__ == "__main__":
    # Test nhanh cấu trúc mô hình
    model = AmodalSwinUNet()
    # Tạo dữ liệu giả: 2 ảnh, 4 kênh, 224x224
    dummy_input = torch.randn(2, 4, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Kiến trúc U-Net đã OK!")
    print(f"Đầu vào: {dummy_input.shape}")
    print(f"Đầu ra: {output.shape} (Phải là 2x1x224x224)")
