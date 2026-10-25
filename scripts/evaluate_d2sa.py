"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL SWIN-UNET TRÊN D2SA (ROW 4 — CẤU HÌNH CHÍNH)
===================================================================================
Script đánh giá mô hình Amodal Shape Prediction trên D2S-Amodal validation set.
Cấu hình: Row 4 (Main Config):
- Input: 5 kênh (RGB + Visible Mask + Edge Mask)
- Category Embedding: 60 lớp
- Spatial Attention: Tắt
- Metrics: mIoU, Dice, Precision, Recall, Invisible mIoU (trên vùng bị che)

Chạy: python scripts/evaluate_d2sa.py --checkpoint checkpoints/d2sa/amodal_shape_prediction_main_config/swin_amodal_epoch_30.pth
===================================================================================
"""

import os
import sys
import json
import argparse

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Đảm bảo đường dẫn import scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset_d2sa import AmodalDatasetD2SA
from model import AmodalSwinUNet


def build_transform(resize_dim=224):
    return A.Compose([
        A.Resize(resize_dim, resize_dim),
    ])


def calculate_metrics(pred_binary, target_binary, visible_binary):
    """
    Tính toán metrics đánh giá cho một ảnh:
    - IoU (Toàn bộ vật thể)
    - Dice
    - Precision
    - Recall
    - Invisible IoU (Chỉ tính trên vùng bị che khuất)
    """
    # Vùng bị che khuất trong ground truth: target AND NOT visible
    occluded_gt = np.logical_and(target_binary == 1, visible_binary == 0).astype(np.uint8)
    has_occlusion = np.sum(occluded_gt) > 0

    # Metrics trên toàn bộ Amodal mask
    intersection = np.logical_and(pred_binary == 1, target_binary == 1).sum()
    union = np.logical_or(pred_binary == 1, target_binary == 1).sum()
    iou = intersection / union if union > 0 else (1.0 if np.sum(target_binary) == 0 else 0.0)

    pred_sum = np.sum(pred_binary)
    target_sum = np.sum(target_binary)
    dice = (2.0 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else 1.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / target_sum if target_sum > 0 else 0.0

    # Invisible IoU: chỉ xét trên vùng bị che
    if has_occlusion:
        pred_occluded = np.logical_and(pred_binary == 1, visible_binary == 0).astype(np.uint8)
        inv_intersection = np.logical_and(pred_occluded == 1, occluded_gt == 1).sum()
        inv_union = np.logical_or(pred_occluded == 1, occluded_gt == 1).sum()
        invisible_iou = inv_intersection / inv_union if inv_union > 0 else 0.0
    else:
        invisible_iou = None

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "invisible_iou": float(invisible_iou) if invisible_iou is not None else 0.0,
        "has_occlusion": bool(has_occlusion)
    }


def evaluate(args):
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🔍 Đánh giá Row 4 D2SA trên thiết bị: {device} | Single-Image Evaluation")

    transform = build_transform(args.resize)
    dataset = AmodalDatasetD2SA(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )

    if args.subset_size and args.subset_size < len(dataset):
        print(f"⚠️ Chế độ Subset: Chỉ đánh giá trên {args.subset_size}/{len(dataset)} mẫu!")
        dataset.annotations = dataset.annotations[:args.subset_size]

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == "cuda" else False
    )

    # Khởi tạo mô hình
    model = AmodalSwinUNet(num_classes=args.num_classes).to(device)

    # Nạp checkpoint
    print(f"📦 Đang nạp checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    raw_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "inv_iou": 0.0}
    occluded_count = 0
    per_sample_metrics = []

    print("📊 Bắt đầu chấm điểm trên tập D2SA validation...")
    with torch.no_grad():
        for idx, (inputs, targets, occluded_region, class_ids) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs = inputs.to(device, non_blocking=True)
            targets_tensor = targets.unsqueeze(1).float().to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            visible_tensor = inputs[:, 3:4, :, :].float().to(device, non_blocking=True)

            outputs = model(inputs, class_ids)
            probs = torch.sigmoid(outputs)
            preds_binary = (probs > args.threshold).cpu().numpy().squeeze().astype(np.uint8)
            targets_np = targets.squeeze().numpy().astype(np.uint8)
            visible_np = visible_tensor.cpu().numpy().squeeze().astype(np.uint8)

            res = calculate_metrics(preds_binary, targets_np, visible_np)
            res["sample_index"] = idx

            total_metrics["iou"] += res["iou"]
            total_metrics["dice"] += res["dice"]
            total_metrics["precision"] += res["precision"]
            total_metrics["recall"] += res["recall"]

            if res["has_occlusion"]:
                total_metrics["inv_iou"] += res["invisible_iou"]
                occluded_count += 1

            per_sample_metrics.append(res)

    n_samples = len(loader)
    m_iou = (total_metrics["iou"] / n_samples) * 100
    m_dice = (total_metrics["dice"] / n_samples) * 100
    m_precision = (total_metrics["precision"] / n_samples) * 100
    m_recall = (total_metrics["recall"] / n_samples) * 100
    m_inv_iou = (total_metrics["inv_iou"] / occluded_count * 100) if occluded_count > 0 else 0.0

    print("\n" + "=" * 60)
    print("📈 KẾT QUẢ ĐÁNH GIÁ D2SA (ROW 4 — MAIN CONFIG)")
    print("=" * 60)
    print(f"Tổng số mẫu:           {n_samples}")
    print(f"Số mẫu có che khuất:   {occluded_count} ({occluded_count/n_samples*100:.1f}%)")
    print(f"mIoU (Toàn bộ amodal): {m_iou:.2f}%")
    print(f"Dice Coefficient:      {m_dice:.2f}%")
    print(f"Precision:             {m_precision:.2f}%")
    print(f"Recall:                {m_recall:.2f}%")
    print(f"Invisible mIoU:        {m_inv_iou:.2f}% (Chỉ vùng bị che)")
    print("=" * 60)

    results = {
        "dataset": args.ann_file,
        "checkpoint": args.checkpoint,
        "config": "Row 4: 5ch + Category Embedding (60) + OccLoss + No Spatial",
        "total_samples": n_samples,
        "occluded_samples": occluded_count,
        "summary_metrics": {
            "mIoU": float(m_iou),
            "dice": float(m_dice),
            "precision": float(m_precision),
            "recall": float(m_recall),
            "invisible_mIoU": float(m_inv_iou),
        },
        "per_sample_metrics": per_sample_metrics,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Kết quả chi tiết đã lưu tại: {args.output}")

    return results


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="Đánh giá mô hình Amodal Swin-UNet trên D2SA (Row 4)")
    parser.add_argument("--img-dir", type=str, default=os.path.join(root_dir, "data", "D2SA", "images"), help="Thư mục chứa ảnh D2SA")
    parser.add_argument("--ann-file", type=str, default=os.path.join(root_dir, "data", "D2SA", "D2S_amodal_validation.json"), help="Annotation validation JSON")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(root_dir, "checkpoints", "d2sa", "amodal_shape_prediction_main_config", "swin_amodal_epoch_30.pth"), help="Đường dẫn checkpoint")
    parser.add_argument("--num-classes", type=int, default=60, help="Số lớp D2SA")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize ảnh")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng nhị phân hóa mask")
    parser.add_argument("--num-workers", type=int, default=2, help="Số worker nạp dữ liệu")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--output", type=str, default=os.path.join(root_dir, "results", "d2sa", "row4_eval.json"), help="Đường dẫn lưu file JSON kết quả")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu đánh giá (0: toàn bộ)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
