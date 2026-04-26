"""
Enhanced Training Script với:
1. Balanced sampling (oversample occluded samples)
2. Advanced loss functions (FocalOcclusionLoss, tuned weights)
3. Experiment tracking (so sánh 5x vs 10x vs 15x vs 20x)

Chạy: python src/train_balanced.py --loss-type focal --occlusion-weight 15 --oversample-ratio 2.0
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
import json

from dataset import AmodalDataset
from model import AmodalSwinUNet
from advanced_loss import (
    OcclusionAwareLoss,
    FocalOcclusionLoss,
    OcclusionFocalLoss,
    create_balanced_dataloader,
)


def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Đang chạy trên thiết bị: {DEVICE}")

    # === SETUP DIRECTORIES ===
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # === LOAD DATASET ===
    print("\n📂 Chuẩn bị dataset...")
    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    train_dataset = AmodalDataset(
        img_dir=os.path.join(project_root, "data/train2014"),
        ann_file=os.path.join(
            project_root, "data/annotations/COCO_amodal_train2014.json"
        ),
        transform=train_transform,
    )

    # === BALANCED DATALOADER ===
    if args.use_balanced_sampling:
        print(
            f"\n⚖️  Tạo balanced sampler (oversample_ratio={args.oversample_ratio}x)..."
        )
        train_loader = create_balanced_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            occlusion_threshold=args.occlusion_threshold,
            oversample_ratio=args.oversample_ratio,
            use_weighted_sampler=True,
        )
    else:
        print("\n📊 Sử dụng random sampling...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # === MODEL ===
    print("\n🧠 Tải model...")
    model = AmodalSwinUNet().to(DEVICE)

    if args.resume_epoch > 0:
        weight_path = os.path.join(
            args.checkpoint_dir, f"swin_amodal_epoch_{args.resume_epoch}.pth"
        )
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print(f"🔄 Tải checkpoint từ epoch {args.resume_epoch}")
        else:
            print(f"⚠️  Checkpoint {weight_path} không tìm thấy, bắt đầu từ đầu")
            args.resume_epoch = 0

    # === LOSS FUNCTION ===
    print(f"\n📉 Sử dụng loss type: {args.loss_type}")
    if args.loss_type == "original":
        criterion = OcclusionAwareLoss(occlusion_weight=args.occlusion_weight)
        loss_name = f"OcclusionAwareLoss({args.occlusion_weight}x)"
    elif args.loss_type == "focal":
        criterion = FocalOcclusionLoss(
            alpha_occlusion=args.occlusion_weight, gamma=args.focal_gamma
        )
        loss_name = (
            f"FocalOcclusionLoss({args.occlusion_weight}x, gamma={args.focal_gamma})"
        )
    elif args.loss_type == "combo":
        criterion = OcclusionFocalLoss(
            alpha_occlusion=args.occlusion_weight,
            gamma=args.focal_gamma,
            use_focal=True,
        )
        loss_name = (
            f"OcclusionFocalLoss({args.occlusion_weight}x, gamma={args.focal_gamma})"
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # === OPTIMIZER & SCHEDULER ===
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === TRAINING LOOP ===
    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN")
    print(f"   Loss Function: {loss_name}")
    print(f"   Balanced Sampling: {args.use_balanced_sampling}")
    if args.use_balanced_sampling:
        print(f"   Occlusion Threshold: {100*args.occlusion_threshold:.0f}%")
        print(f"   Oversample Ratio: {args.oversample_ratio}x")
    print(f"   Epochs: {args.resume_epoch + 1} → {args.epochs}")
    print("=" * 70)

    training_log = {
        "config": {
            "loss_type": args.loss_type,
            "occlusion_weight": args.occlusion_weight,
            "balanced_sampling": args.use_balanced_sampling,
            "occlusion_threshold": args.occlusion_threshold,
            "oversample_ratio": args.oversample_ratio,
        },
        "epochs": [],
    }

    best_loss = float("inf")

    for epoch in range(args.resume_epoch, args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
        )

        for batch_idx, (inputs, targets, occluded, class_ids) in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)
            occluded = occluded.unsqueeze(1).float().to(DEVICE)
            class_ids = class_ids.to(DEVICE)

            # Forward pass
            outputs = model(inputs, class_ids)
            loss = criterion(outputs, targets, occluded)

            # Gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix(
                loss=loss.item() * args.gradient_accumulation_steps
            )

        # Scheduler step
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        epoch_log = {"epoch": epoch + 1, "loss": avg_loss, "lr": current_lr}
        training_log["epochs"].append(epoch_log)

        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Save checkpoint
        if avg_loss < best_loss or (epoch + 1) % args.save_every == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                is_best = True
            else:
                is_best = False

            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"swin_amodal_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)

            marker = "⭐ BEST" if is_best else "💾"
            print(f"{marker} Checkpoint saved: {checkpoint_path}")

        print()

    # === SAVE TRAINING LOG ===
    log_path = os.path.join(
        args.results_dir, f"training_log_{args.loss_type}_w{args.occlusion_weight}.json"
    )
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n💾 Training log saved: {log_path}")

    print("\n🎉 HUẤN LUYỆN HOÀN TẤT!")


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện Amodal Model với advanced loss functions"
    )

    # Model arguments
    parser.add_argument(
        "--resume-epoch", type=int, default=0, help="Epoch để resume training"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Tổng số epoch")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Số workers cho DataLoader"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )

    # Loss function arguments
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["original", "focal", "combo"],
        default="combo",
        help="Loại loss function",
    )
    parser.add_argument(
        "--occlusion-weight",
        type=float,
        default=15.0,
        help="Trọng số cho occlusion regions",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0, help="Gamma parameter cho Focal Loss"
    )

    # Balanced sampling arguments
    parser.add_argument(
        "--use-balanced-sampling",
        action="store_true",
        default=True,
        help="Sử dụng balanced sampling",
    )
    parser.add_argument(
        "--no-balanced-sampling",
        dest="use_balanced_sampling",
        action="store_false",
        help="Không sử dụng balanced sampling",
    )
    parser.add_argument(
        "--occlusion-threshold",
        type=float,
        default=0.1,
        help="Threshold để xác định 'occluded' sample (0-1)",
    )
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=2.0,
        help="Bao nhiêu lần oversample occluded samples",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="../checkpoints",
        help="Thư mục lưu checkpoint",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Thư mục lưu kết quả training",
    )
    parser.add_argument(
        "--save-every", type=int, default=5, help="Lưu checkpoint mỗi N epoch"
    )

    args = parser.parse_args()

    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.join(project_root, args.checkpoint_dir)
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(project_root, args.results_dir)

    train(args)


if __name__ == "__main__":
    main()
