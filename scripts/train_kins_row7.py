"""
===================================================================================
HUẤN LUYỆN ROW 7 (SG-GBA SWIN-UNET) TRÊN KINS (20 CLASSES)
===================================================================================
Cấu hình:
- Đầu vào: 5 kênh (RGB 3ch + Visible 1ch + Edge 1ch) + Category ID
- Category Embedding: 20 lớp KINS
- Bottleneck Attention: SGGatedBottleneck (FiLM + Conv2d 3x3 + Learnable Gate alpha)
- Loss: Occlusion-Aware Loss (5x weight) + Soft Dice Loss
- Optimizer: AdamW, lr=1e-4 (hoặc 5e-5 cho fine-tune), CosineAnnealingLR
- Checkpoint: Lưu vào checkpoints/kins/row7_sg_gba/
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

# Thêm đường dẫn scripts vào sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset_kins import KINSDataset
from model_kins_row7 import AmodalSwinUNetRow7KINS


class OcclusionAwareLoss(nn.Module):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu huấn luyện Row 7 KINS (SG-GBA) trên thiết bị: {device}")

    # Transform
    train_transform = A.Compose([
        A.Resize(args.resize, args.resize),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    print(f"📂 Đang khởi tạo DataLoader KINS từ {args.img_dir}...")
    dataset = KINSDataset(img_dir=args.img_dir, ann_file=args.ann_file, transform=train_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    EXPECTED_PARAMS = 34_813_038
    model = AmodalSwinUNetRow7KINS(pretrained=True, num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {total_params:,} (Kỳ vọng: {EXPECTED_PARAMS:,})")
    assert total_params == EXPECTED_PARAMS, f"Lệch tham số Row 7 KINS: {total_params} != {EXPECTED_PARAMS}"

    os.makedirs(args.save_dir, exist_ok=True)

    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"🔄 Đã nạp checkpoint: {args.resume_checkpoint}")
        if args.resume_epoch > 0:
            start_epoch = args.resume_epoch
            for _ in range(start_epoch):
                scheduler.step()

    print(f"\n🔥 HUẤN LUYỆN ROW 7 KINS: Epoch {start_epoch + 1} -> {args.epochs} 🔥")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for inputs, targets, occluded, class_ids in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(device, non_blocking=True)
            occluded = occluded.unsqueeze(1).float().to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs, class_ids)
                loss = criterion(outputs, targets, occluded)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        gate_alpha = float(model.sg_bottleneck.alpha.item())
        print(f"✅ Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Gate Alpha: {gate_alpha:.4f}")

        # Lưu checkpoint định kỳ và atomic last.pth
        ckpt_path = os.path.join(args.save_dir, f"swin_amodal_kins_row7_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)

        last_path = os.path.join(args.save_dir, "last.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, last_path)

    print("\n🎉 HOÀN THÀNH HUẤN LUYỆN ROW 7 KINS!")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Row 7 KINS (SG-GBA)")
    parser.add_argument("--img-dir", type=str, default="/content/kitti_data/training/image_2", help="Thư mục ảnh KITTI")
    parser.add_argument("--ann-file", type=str, default="/content/kins_data/update_train_2020.json", help="File annotation KINS train")
    parser.add_argument("--save-dir", type=str, default="checkpoints/kins/row7_sg_gba", help="Thư mục lưu weights")
    parser.add_argument("--num-classes", type=int, default=20, help="Số lớp đối tượng KINS (20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (64 cho A100)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate ban đầu")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize ảnh")
    parser.add_argument("--num-workers", type=int, default=8, help="Số worker nạp dữ liệu")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Đường dẫn checkpoint để resume")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Epoch tiếp tục")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
