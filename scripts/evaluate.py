"""
===================================================================================
ĐÁN GIÁ MÔ HÌNH AMODAL PREDICTION
===================================================================================
Script đánh giá hiệu suất mô hình trên validation set.

Metricsused:
- IoU (Intersection over Union): Tính lên toàn bộ mask amodal
- Dice Coefficient: F1-score cho segmentation
- Invisible IoU: IoU chỉ tính trên vùng bị che khuất (occlusion region)

Đầu ra:
- Tổng hợp kết quả (mIoU, Dice, Invisible mIoU)
- Per-sample metrics (lưu để phân tích failure cases)

Chạy: python src/evaluate.py --img-dir data/val2014 --ann-file data/annotations/COCO_amodal_val2014.json
===================================================================================
"""

import argparse
import json
import os

import albumentations as A
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AmodalSwinUNet
from dataset import AmodalDataset


def calculate_metrics(pred_logits, target, visible, threshold=0.5):
    """
    Tính toán các metrics (IoU, Dice, Invisible IoU).
    
    Args:
        pred_logits: Dự đoán logit từ mô hình [B, 1, H, W]
        target: Amodal mask nhãn [B, 1, H, W]
        visible: Visible mask [B, 1, H, W]
        threshold: Ngưỡng để chuyển logit thành binary mask
    
    Returns:
        Tuple:
        - iou: IoU cho toàn bộ mask [B]
        - dice: Dice coefficient [B]
        - inv_iou: IoU cho vùng bị che khuất [B]
        - valid_mask: Mask xác định các sample có occlusion [B]
    """
    # Chuyển logit thành binary prediction (0 hoặc 1)
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    # ─────────────────────────────────────────────────────────────
    # Tính IoU trên toàn bộ mask (Overall IoU)
    # ─────────────────────────────────────────────────────────────
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # ─────────────────────────────────────────────────────────────
    # Tính Dice Coefficient
    # ─────────────────────────────────────────────────────────────
    dice = (2.0 * intersection + 1e-6) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
    )

    # ─────────────────────────────────────────────────────────────
    # Tính IoU cho vùng bị che khuất (Invisible IoU)
    # ─────────────────────────────────────────────────────────────
    # Vùng bị che khuất = amodal - visible (nơi mà amodal có nhưng visible không)
    invisible_target = torch.clamp(target - visible, min=0.0)
    pred_invisible = pred * (invisible_target > 0).float()

    inv_intersection = (pred_invisible * invisible_target).sum(dim=(2, 3))
    inv_union = (
        pred_invisible.sum(dim=(2, 3))
        + invisible_target.sum(dim=(2, 3))
        - inv_intersection
    )
    # valid_mask: 1 nếu có occlusion, 0 nếu không
    valid_mask = (invisible_target.sum(dim=(2, 3)) > 0).float()
    inv_iou = (inv_intersection + 1e-6) / (inv_union + 1e-6)

    return iou, dice, inv_iou, valid_mask


def build_transform(resize):
    """
    Xây dựng augmentation pipeline cho evaluation.
    
    Lưu ý: Evaluation không dùng augmentation, chỉ resize
    
    Args:
        resize: Kích thước resize
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([A.Resize(resize, resize)])


def evaluate(args):
    """
    Hàm chính để đánh giá mô hình.
    
    Args:
        args: Argparse arguments từ parse_args()
    """
    # Chọn thiết bị
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🔍 Đánh giá trên thiết bị: {device}")

    # ─────────────────────────────────────────────────────────────
    # CHUẨN BỊ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────
    transform = build_transform(args.resize)
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ─────────────────────────────────────────────────────────────
    # NẠP MÔ HÌNH
    # ─────────────────────────────────────────────────────────────
    model = AmodalSwinUNet(num_classes=91).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()  # Bật chế độ evaluation

    # ─────────────────────────────────────────────────────────────
    # VÒNG LẶP ĐÁNH GIÁ
    # ─────────────────────────────────────────────────────────────
    total_iou = 0.0
    total_dice = 0.0
    total_inv_iou = 0.0
    total_valid_inv = 0.0

    # Lưu per-sample metrics để phân tích sau này
    per_sample_metrics = []

    print("📊 Tính toán metrics từng pixel... Xin chờ!")
    with torch.no_grad():
        for inputs, targets, occluded_region, class_ids in tqdm(
            loader, desc="Evaluating"
        ):
            # Di chuyển dữ liệu lên device
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            visible_masks = inputs[:, 3:4, :, :].float()  # Kênh 4 là visible mask
            class_ids = class_ids.to(device)

            # Dự đoán
            outputs = model(inputs, class_ids)
            iou, dice, inv_iou, valid_mask = calculate_metrics(
                outputs, targets, visible_masks, threshold=args.threshold
            )

            # Cộng dồn metrics
            total_iou += iou.sum().item()
            total_dice += dice.sum().item()
            total_inv_iou += (inv_iou * valid_mask).sum().item()
            total_valid_inv += valid_mask.sum().item()

            # Lưu per-sample metrics
            for i in range(iou.shape[0]):
                per_sample_metrics.append(
                    {
                        "iou": iou[i].item(),
                        "dice": dice[i].item(),
                        "invisible_iou": (
                            inv_iou[i].item() if valid_mask[i].item() > 0 else -1.0
                        ),
                        "has_occlusion": valid_mask[i].item() > 0,
                    }
                )

    # ─────────────────────────────────────────────────────────────
    # TÍNH TRUNG BÌNH & IN KẾT QUẢ
    # ─────────────────────────────────────────────────────────────
    n_samples = len(dataset)
    m_iou = total_iou / n_samples if n_samples > 0 else 0.0
    m_dice = total_dice / n_samples if n_samples > 0 else 0.0
    m_inv_iou = total_inv_iou / total_valid_inv if total_valid_inv > 0 else 0.0

    print("\n" + "=" * 60)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"📂 Dataset           : {args.ann_file}")
    print(f"📦 Checkpoint        : {args.checkpoint}")
    print(f"📊 Tổng số mẫu      : {n_samples}")
    print(f"🎯 Overall mIoU      : {m_iou * 100:.2f}%")
    print(f"🎲 Dice Coefficient  : {m_dice * 100:.2f}%")
    print(f"👁️  Invisible mIoU    : {m_inv_iou * 100:.2f}%")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # LƯU KẾT QUẢ
    # ─────────────────────────────────────────────────────────────
    results = {
        "dataset": args.ann_file,
        "checkpoint": args.checkpoint,
        "samples": n_samples,
        "overall_mIoU": m_iou,
        "dice": m_dice,
        "invisible_mIoU": m_inv_iou,
        "threshold": args.threshold,
        "resize": args.resize,
        "device": str(device),
        "per_sample_metrics": per_sample_metrics,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Kết quả lưu tại: {args.output}")

    return results


def parse_args():
    """Phân tích command-line arguments."""
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Amodal Segmentation")
    parser.add_argument(
        "--img-dir",
        type=str,
        default="../data/val2014",
        help="Thư mục chứa ảnh validation",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="../data/annotations/COCO_amodal_val2014.json",
        help="File annotation COCO-Amodal",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/swin_amodal_epoch_30.pth",
        help="Đường dẫn checkpoint mô hình",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Kích thước batch"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Số worker DataLoader"
    )
    parser.add_argument(
        "--resize", type=int, default=224, help="Kích thước resize input"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng sigmoid để tạo binary mask",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Thiết bị: auto, cpu, hoặc cuda"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Lưu kết quả ra file JSON"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
