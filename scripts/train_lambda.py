"""
===================================================================================
HUẤN LUYỆN AMODAL SWIN-UNET VỚI NHIỀU TRỌNG SỐ OCCLUSION KHÁC NHAU
===================================================================================
Script huấn luyện mô hình Amodal Shape Prediction trên COCO-Amodal dataset.

Tính năng:
- Sử dụng loss function đặc biệt cho occlusion
- Huấn luyện thử nghiệm với nhiều occlusion_weight khác nhau (1, 3, 5, 7, 10)
- Mỗi weight huấn luyện 10 epochs.

Chạy: python scripts/train_lambda.py
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
    
    Args:
        occlusion_weight: Hệ số nhân trọng lượng cho vùng bị che khuất
    """
    def __init__(self, occlusion_weight=5.0):
        super().__init__()
        # BCE loss tính từng pixel riêng biệt (reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        """
        Tính loss với tập trọng số khác nhau cho vùng bị che và không bị che.
        """
        # Tính BCE loss cho từng pixel
        bce_loss = self.bce(pred, target)
        
        # Tạo ma trận trọng số: mặc định 1, ở vùng occlusion là occlusion_weight
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
    Hàm chính để huấn luyện mô hình với nhiều lambda (occlusion_weight).
    """
    
    # ─────────────────────────────────────────────────────────────────
    # CẤU HÌNH HUẤN LUYỆN DÀNH CHO A100
    # ─────────────────────────────────────────────────────────────────
    BATCH_SIZE = 64         # A100 có VRAM lớn (40/80GB) nên có thể đẩy batch size lên cao (64 hoặc 128)
    ACCUMULATION_STEPS = 1  # Không cần gradient accumulation nếu batch size đã đủ lớn
    EPOCHS = 10
    LEARNING_RATE = 2e-4    # Có thể tăng LR một chút so với 1e-4 do batch size lớn hơn
    LAMBDA_WEIGHTS = [1, 3, 5, 7, 10]

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

    print("📂 Chuẩn bị DataLoader với data augmentation (Tối ưu cho A100)...")
    train_dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=train_transform)
    # Tăng num_workers và dùng pin_memory=True để tránh nghẽn CPU khi nạp dữ liệu cho GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,       # Điều chỉnh theo số CPU core (thường 8-16 là tốt trên server)
        pin_memory=True      # Giúp truyền dữ liệu CPU -> GPU nhanh hơn
    )

    # Tạo thư mục lưu checkpoint
    os.makedirs("../checkpoints", exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # VÒNG LẶP HUẤN LUYỆN CHO TỪNG LAMBDA
    # ─────────────────────────────────────────────────────────────────
    # Khởi tạo scaler cho Mixed Precision Training (AMP) giúp tăng tốc vượt trội trên A100
    scaler = torch.cuda.amp.GradScaler()

    for weight in LAMBDA_WEIGHTS:
        print(f"\n{'='*70}")
        print(f"🔥 BẮT ĐẦU HUẤN LUYỆN VỚI OCCLUSION WEIGHT = {weight} 🔥")
        print(f"{'='*70}")
        
        # ─────────────────────────────────────────────────────────────────
        # KHỞI TẠO MÔ HÌNH VÀ OPTIMIZER CHO MỖI TRỌNG SỐ
        # ─────────────────────────────────────────────────────────────────
        # Khởi tạo mô hình mới từ đầu để đánh giá độc lập từng weight
        model = AmodalSwinUNet().to(DEVICE)
        
        criterion = OcclusionAwareLoss(occlusion_weight=weight)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # Learning rate scheduler: giảm LR theo Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        for epoch in range(EPOCHS):
            model.train()  # Bật chế độ training (dropout, batch norm, ...)
            total_loss = 0
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True giúp tối ưu bộ nhớ

            # Thanh tiến trình TQDM
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Weight {weight} | Epoch {epoch+1}/{EPOCHS}"
            )

            for i, (inputs, targets, occluded, class_ids) in progress_bar:
                # Di chuyển dữ liệu lên GPU
                # Dùng non_blocking=True đi kèm với pin_memory=True để truyền dữ liệu bất đồng bộ
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.unsqueeze(1).float().to(DEVICE, non_blocking=True)
                occluded = occluded.unsqueeze(1).float().to(DEVICE, non_blocking=True)
                class_ids = class_ids.to(DEVICE, non_blocking=True)

                # Forward pass sử dụng Automatic Mixed Precision
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, class_ids) 
                    loss = criterion(outputs, targets, occluded)
                    loss = loss / ACCUMULATION_STEPS 
                
                # Backward với scaler
                scaler.scale(loss).backward()

                # Cập nhật trọng số
                if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # Cộng loss (nhân lại với ACCUMULATION_STEPS để trả về giá trị thực)
                total_loss += loss.item() * ACCUMULATION_STEPS
                # Cập nhật thanh tiến trình
                progress_bar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

            # Cập nhật learning rate theo schedule
            scheduler.step()

            # Tính loss trung bình của epoch
            avg_loss = total_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"✅ Weight {weight} | Epoch {epoch+1} hoàn tất | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

            # Lưu checkpoint
            save_path = f"../checkpoints/swin_amodal_lambda_{weight}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Checkpoint lưu tại: {save_path}\n")


if __name__ == "__main__":
    train()
