import torch
import torch.nn as nn
import timm


class AmodalSwinUNet(nn.Module):
    def __init__(self):
        super().__init__()
        print("Đang khởi tạo bộ não Swin Transformer 4-Channel...")

        # 1. Tải mô hình Swin Transformer gốc (đã train sẵn siêu xịn trên ImageNet)
        # Dùng bản 'tiny' cho nhẹ máy dễ train, patch_size=4
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True
        )

        # 2. PHẪU THUẬT LỚP ĐẦU VÀO (3 KÊNH -> 4 KÊNH)
        # Lấy lớp Convolution đầu tiên của Swin ra
        old_conv = self.encoder.patch_embed.proj

        # Tạo một lớp Conv mới y hệt, nhưng nhận in_channels = 4
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        # COPY TRỌNG SỐ (Bí kíp để AI không bị ngu đi)
        with torch.no_grad():
            # Giữ nguyên kiến thức 3 kênh RGB gốc
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Kênh thứ 4 (Visible Mask), ta lấy trung bình cộng của 3 kênh kia làm giá trị khởi tạo
            new_conv.weight[:, 3, :, :] = old_conv.weight.mean(dim=1)
            new_conv.bias = old_conv.bias

        # Tráo lớp mới vào mô hình, vứt lớp cũ đi
        self.encoder.patch_embed.proj = new_conv

        # Xóa lớp classification head gốc (vì mình làm Segmentation chứ không phân loại ảnh)
        self.encoder.head = nn.Identity()

        # 3. XÂY DỰNG DECODER (Giải mã ngược từ não AI ra lại bức ảnh Mask)
        # Swin Tiny sẽ nén ảnh 256x256 xuống thành một ma trận đặc cỡ 8x8 với 768 features.
        # Tụi mình cần cái Decoder để phóng to nó ngược lại thành kích thước 256x256
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Phóng to x4 lần (lên 32x32)
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Phóng to x8 lần nữa (lên đúng 256x256)
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            # Ép về 1 kênh duy nhất: Bức ảnh Amodal Mask trắng đen!
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        # 1. Rút trích đặc trưng bằng Swin
        features = self.encoder.forward_features(x)

        # 2. Định dạng lại: Từ (Batch, H, W, Channels) -> (Batch, Channels, H, W)
        # Bằng cách chuyển vị trí Channel (số 3) lên trước H và W (số 1 và 2)
        features = features.permute(0, 3, 1, 2)

        # 3. Bơm qua ống Decoder
        out = self.decoder(features)
        return out
