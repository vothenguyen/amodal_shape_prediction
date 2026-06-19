"""
===================================================================================
HUẤN LUYỆN AMODAL SWIN-UNET
===================================================================================
Script huấn luyện mô hình Amodal Shape Prediction trên COCO-Amodal dataset.

Tính năng:
- Sử dụng loss function BCEWithLogitsLoss thông thường
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
    RESUME_EPOCH = 26  # Tiếp tục từ epoch 26 nếu có checkpoint
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
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print(f"\n🔄 Tiếp tục từ Epoch {RESUME_EPOCH}: Đã nạp trọng số từ checkpoint!")
        else:
            print(f"\n⚠️ Không tìm thấy checkpoint tại {weight_path}. Tự động bắt đầu từ Epoch 0!")
            RESUME_EPOCH = 0

    # ─────────────────────────────────────────────────────────────────
    # LOSS FUNCTION & OPTIMIZER
    # ─────────────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: giảm LR theo Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Tạo thư mục lưu checkpoint
    os.makedirs("../checkpoints", exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # VÒNG LẶP HUẤN LUYỆN CHÍNH
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

        for i, (inputs, targets, occluded, _) in progress_bar:
            # Di chuyển dữ liệu lên GPU
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)  # Thêm chiều kênh

            # Forward pass: tính dự đoán
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
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
