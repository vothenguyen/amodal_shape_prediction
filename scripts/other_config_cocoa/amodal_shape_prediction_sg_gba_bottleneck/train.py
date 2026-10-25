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
- Tối ưu hóa A100: batch_size=16, num_workers=8, pin_memory=True, PyTorch AMP (FP16)
- Hỗ trợ resume mượt mà từ epoch bất kỳ (--resume-epoch N)

Chạy: python scripts/other_config_cocoa/amodal_shape_prediction_sg_gba_bottleneck/train.py
===================================================================================
"""

import os
import sys
import argparse
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


def train(args):
    # ─────────────────────────────────────────────────────────────────
    # THIẾT LẬP SIÊU THAM SỐ
    # ─────────────────────────────────────────────────────────────────
    BATCH_SIZE = args.batch_size
    ACCUMULATION_STEPS = args.accumulation_steps
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    RESUME_EPOCH = args.resume_epoch
    EXPECTED_PARAMS = 34_867_566

    DEVICE = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    USE_AMP = args.amp and (DEVICE.type == "cuda")

    print(f"🚀 Huấn luyện Row 7 COCOA (SG-GBA) trên thiết bị: {DEVICE} (AMP: {USE_AMP})")
    print(f"⚡ Batch Size: {BATCH_SIZE} | Acc Steps: {ACCUMULATION_STEPS} | Workers: {args.num_workers}")

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None
    )

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
            ckpt = torch.load(weight_path, map_location=DEVICE)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"\n🔄 Tiếp tục từ Epoch {RESUME_EPOCH}: Đã nạp trọng số từ checkpoint: {weight_path}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy checkpoint tại {weight_path}, sẽ huấn luyện từ đầu!")
            RESUME_EPOCH = 0

    # ─────────────────────────────────────────────────────────────────
    # LOSS FUNCTION & OPTIMIZER
    # ─────────────────────────────────────────────────────────────────
    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Tua nhanh scheduler nếu resume
    if RESUME_EPOCH > 0:
        for _ in range(RESUME_EPOCH):
            scheduler.step()
        print(f"⚡ Fast-forward LR Scheduler về Epoch {RESUME_EPOCH} (LR: {scheduler.get_last_lr()[0]:.2e})")

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN ROW 7 (SG-GBA): Epoch {RESUME_EPOCH + 1} → {EPOCHS} 🔥")
    print("=" * 70)

    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for i, (inputs, targets, occluded, class_ids) in progress_bar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(DEVICE, non_blocking=True)
            occluded = occluded.unsqueeze(1).float().to(DEVICE, non_blocking=True)
            class_ids = class_ids.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs, class_ids)
                loss = criterion(outputs, targets, occluded)
                if ACCUMULATION_STEPS > 1:
                    loss = loss / ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
                if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            actual_loss = loss.item() * (ACCUMULATION_STEPS if ACCUMULATION_STEPS > 1 else 1)
            total_loss += actual_loss
            progress_bar.set_postfix(
                loss=f"{actual_loss:.4f}",
                alpha=f"{float(model.sg_bottleneck.alpha.item()):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        current_alpha = float(model.sg_bottleneck.alpha.item())
        print(f"✅ Epoch {epoch+1} hoàn tất | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Alpha: {current_alpha:.5f}")

        save_path = os.path.join(ckpt_dir, f"swin_amodal_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint lưu tại: {save_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Row 7 COCOA (SG-GBA)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (mặc định 16 cho A100)")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Số bước tích lũy gradient")
    parser.add_argument("--epochs", type=int, default=30, help="Tổng số epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học AdamW")
    parser.add_argument("--num-workers", type=int, default=8, help="Số luồng CPU nạp dữ liệu (mặc định 8)")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Epoch muốn resume (0: chạy từ đầu)")
    parser.add_argument("--amp", action="store_true", default=True, help="Bật Mixed Precision FP16")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
