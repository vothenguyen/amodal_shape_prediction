"""
===================================================================================
HUẤN LUYỆN AMODAL SWIN-UNET TRÊN COCOA (ROW 7 — PROPOSED: SG-GBA)
===================================================================================
Cấu hình: Row 7 trong Ablation Study (Kiến trúc đề xuất mới cho Khóa luận tốt nghiệp):
- Đầu vào: 5 kênh (RGB + Visible Mask + Edge Mask)
- Category Embedding: Bật (91 lớp COCO)
- Bottleneck Attention: SGGatedBottleneck (FiLM điều biến ngữ nghĩa + Chú ý không gian 7x7 + Residual Gate)
- Spatial Attention ở cuối Decoder: Tắt (khắc phục Spatial Attention Paradox)
- Hàm mất mát: Occlusion-Aware Loss (5x weight cho vùng bị che khuất) + Dice Loss
- Effective batch size: 16 (batch_size=4, accumulation_steps=4)
- Optimizer: AdamW (lr=1e-4), Cosine Annealing LR

Chạy: python scripts/other_config_cocoa/amodal_shape_prediction_sg_gba_bottleneck/train.py
===================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Đảm bảo import đúng model.py cục bộ và dataset từ scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from dataset import AmodalDataset
from model import AmodalSwinUNet


class OcclusionAwareLoss(nn.Module):
    """
    Loss function thiết kế riêng cho Amodal Prediction.
    - Phần bị che khuất (occlusion region) khó dự đoán hơn → weight 5x
    - Kết hợp weighted BCE loss + Dice loss
    """
    def __init__(self, occlusion_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        bce_loss = self.bce(pred, target)

        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = self.occlusion_weight

        weighted_bce = (bce_loss * weight_matrix).mean()

        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()


def train():
    # ─────────────────────────────────────────────────────────────────
    # THIẾT LẬP SIÊU THAM SỐ
    # ─────────────────────────────────────────────────────────────────
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    RESUME_EPOCH = 0
    EXPECTED_PARAMS = 34_867_566

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Huấn luyện Row 7 COCOA (SG-GBA) trên thiết bị: {DEVICE}")

    # ─────────────────────────────────────────────────────────────────
    # CHUẨN BỊ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────────
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
    img_dir = os.path.join(ROOT_DIR, "data", "train2014")
    ann_file = os.path.join(ROOT_DIR, "data", "annotations", "COCO_amodal_train2014.json")

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    print("📂 Chuẩn bị DataLoader với data augmentation...")
    train_dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ─────────────────────────────────────────────────────────────────
    # KHỞI TẠO MÔ HÌNH & ASSERTION SỐ LƯỢNG THAM SỐ
    # ─────────────────────────────────────────────────────────────────
    model = AmodalSwinUNet(num_classes=91).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {total_params:,} (Kỳ vọng: {EXPECTED_PARAMS:,})")
    assert total_params == EXPECTED_PARAMS, f"Lệch tham số Row 7 COCOA: {total_params} != {EXPECTED_PARAMS}"

    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "cocoa", "row7_sg_gba_bottleneck")
    os.makedirs(ckpt_dir, exist_ok=True)

    if RESUME_EPOCH > 0:
        weight_path = os.path.join(ckpt_dir, f"swin_amodal_epoch_{RESUME_EPOCH}.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print(f"\n🔄 Tiếp tục từ Epoch {RESUME_EPOCH}: Đã nạp trọng số từ checkpoint!")

    # ─────────────────────────────────────────────────────────────────
    # LOSS FUNCTION & OPTIMIZER
    # ─────────────────────────────────────────────────────────────────
    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN ROW 7 (SG-GBA): Epoch {RESUME_EPOCH + 1} → {EPOCHS} 🔥")
    print("=" * 70)

    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for i, (inputs, targets, occluded, class_ids) in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)
            occluded = occluded.unsqueeze(1).float().to(DEVICE)
            class_ids = class_ids.to(DEVICE)

            outputs = model(inputs, class_ids)
            loss = criterion(outputs, targets, occluded)

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            actual_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += actual_loss
            progress_bar.set_postfix(
                loss=f"{actual_loss:.4f}",
                alpha=f"{float(model.sg_bottleneck.alpha.item()):.4f}"
            )

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        current_alpha = float(model.sg_bottleneck.alpha.item())
        print(f"✅ Epoch {epoch+1} hoàn tất | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Alpha: {current_alpha:.5f}")

        save_path = os.path.join(ckpt_dir, f"swin_amodal_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint lưu tại: {save_path}\n")


if __name__ == "__main__":
    train()
