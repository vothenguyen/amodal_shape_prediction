"""
===================================================================================
HUẤN LUYỆN AMODAL SWIN-UNET
===================================================================================
Script huấn luyện mô hình Amodal Shape Prediction trên COCO-Amodal dataset.

Tính năng:
- Sử dụng loss function đặc biệt cho occlusion (5x weight cho vùng bị che)
- Gradient accumulation để tăng batch size hiệu quả
- Learning rate scheduling (Cosine annealing)
- Progress bar theo dõi training
- Tự động lưu checkpoint sau mỗi epoch

Chạy: python src/train.py
===================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Import các module từ project
from dataset import AmodalDataset
from model import AmodalSwinUNet


class OcclusionAwareLoss(nn.Module):
    """
    Loss function thiết kế riêng cho Amodal Prediction.
    
    Ý tưởng:
    - Phần bị che khuất (occlusion region) khó dự đoán hơn → cần weight cao hơn
    - Kết hợp weighted BCE loss + Dice loss
    - Weight multiplier cho occlusion: 5x (có thể điều chỉnh)
    
    Args:
        occlusion_weight: Hệ số nhân trọng lượng cho vùng bị che khuất (mặc định: 5.0)
    """
    def __init__(self, occlusion_weight=5.0):
        super().__init__()
        # BCE loss tính từng pixel riêng biệt (reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        """
        Tính loss với tập trọng số khác nhau cho vùng bị che và không bị che.
        
        Args:
            pred: Dự đoán logit [B, 1, H, W]
            target: Amodal mask nhãn [B, 1, H, W]
            occluded_region: Vùng bị che khuất [B, 1, H, W] (0 hoặc 1)
        
        Returns:
            Tổng loss (scalar)
        """
        # Tính BCE loss cho từng pixel
        bce_loss = self.bce(pred, target)
        
        # Tạo ma trận trọng số: mặc định 1, ở vùng occlusion là 5x
        weight_matrix = torch.ones_like(target) 
        weight_matrix[occluded_region > 0.5] = self.occlusion_weight
        
        # Áp dụng trọng số vào BCE loss
        weighted_bce = (bce_loss * weight_matrix).mean()

        # Tính Dice loss để tăng cân bằng
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        # Kết hợp hai loss
        return weighted_bce + dice_loss.mean()


def train():
    """
    Hàm chính để huấn luyện mô hình.
    
    Cấu hình huấn luyện:
    - Batch size: 4
    - Gradient accumulation: 4 steps → Batch hiệu quả: 16
    - Epochs: 30
    - Learning rate: 1e-4 với Cosine annealing
    """
    
    # ─────────────────────────────────────────────────────────────────
    # CẤMNH HÌNH HUẤN LUYỆN
    # ─────────────────────────────────────────────────────────────────
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4  # Accumulate gradients 4 lần → Batch ảo 16
    EPOCHS = 30
    RESUME_EPOCH = 20  # Tiếp tục từ epoch 20 nếu có checkpoint
    LEARNING_RATE = 1e-4

    # Chọn thiết bị (GPU nếu có, không thì CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Huấn luyện trên thiết bị: {DEVICE}")

    # ─────────────────────────────────────────────────────────────────
    # CHUẨN BỊ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────────
    img_dir = "../data/train2014"
    ann_file = "../data/annotations/COCO_amodal_train2014.json"

    # Định nghĩa data augmentation cho training
    train_transform = A.Compose([
        A.Resize(224, 224),                                    # Chuẩn hóa kích thước
        A.HorizontalFlip(p=0.5),                               # Lật ngang 50%
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),  # Biến đổi hình học
        A.RandomBrightnessContrast(p=0.2),                     # Thay đổi độ sáng/tương phản
    ])

    print("📂 Chuẩn bị DataLoader với data augmentation...")
    train_dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ─────────────────────────────────────────────────────────────────
    # KHỞI TẠO MÔ HÌNH
    # ─────────────────────────────────────────────────────────────────
    model = AmodalSwinUNet().to(DEVICE)

    # Nếu có checkpoint, tiếp tục từ epoch đó
    if RESUME_EPOCH > 0:
        weight_path = f"../checkpoints/swin_amodal_epoch_{RESUME_EPOCH}.pth"
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"\n🔄 Tiếp tục từ Epoch {RESUME_EPOCH}: Đã nạp trọng số từ checkpoint!")

    # ─────────────────────────────────────────────────────────────────
    # LOSS FUNCTION & OPTIMIZER
    # ─────────────────────────────────────────────────────────────────
    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: giảm LR theo Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Tạo thư mục lưu checkpoint
    os.makedirs("../checkpoints", exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # VÒNG LẶPHUẤN LUYỆN CHÍNH
    # ─────────────────────────────────────────────────────────────────
    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN: Epoch {RESUME_EPOCH + 1} → {EPOCHS} 🔥")
    print("=" * 70)
    
    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()  # Bật chế độ training (dropout, batch norm, ...)
        total_loss = 0
        optimizer.zero_grad()  # Xóa gradient cũ

        # Thanh tiến trình TQDM
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for i, (inputs, targets, occluded, class_ids) in progress_bar:
            # Di chuyển dữ liệu lên GPU
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)  # Thêm chiều kênh
            occluded = occluded.unsqueeze(1).float().to(DEVICE)
            class_ids = class_ids.to(DEVICE)

            # Forward pass: tính dự đoán
            outputs = model(inputs, class_ids) 
            loss = criterion(outputs, targets, occluded)
            
            # Gradient accumulation: chia loss cho số bước tích lũy
            loss = loss / ACCUMULATION_STEPS 
            loss.backward()

            # Cập nhật trọng số mỗi ACCUMULATION_STEPS bước hoặc cuối batch
            if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            # Cộng loss (nhân lại với ACCUMULATION_STEPS để trả về giá trị thực)
            total_loss += loss.item() * ACCUMULATION_STEPS
            # Cập nhật thanh tiến trình
            progress_bar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        # Cập nhật learning rate theo schedule
        scheduler.step()

        # Tính loss trung bình của epoch
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"✅ Epoch {epoch+1} hoàn tất | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Lưu checkpoint
        save_path = f"../checkpoints/swin_amodal_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint lưu tại: {save_path}\n")


if __name__ == "__main__":
    train()