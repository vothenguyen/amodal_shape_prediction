"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL SWIN-UNET (ROW 6 D2SA — FULL CONFIG)
===================================================================================
Cấu hình: Row 6 Ablation Study:
- Đầu vào: 5 kênh (RGB + Visible Mask + Edge Mask)
- Category Embedding: Bật (60 lớp)
- Spatial Attention: Bật (kernel=7)

Chạy: python scripts/other_config_d2sa/amodal_shape_prediction_full_config/evaluate.py
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

# Đảm bảo import đúng model.py cục bộ trước, sau đó mới nạp dataset_d2sa từ scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from dataset_d2sa import AmodalDatasetD2SA
from model import AmodalSwinUNet


def calculate_metrics(pred_binary, target_binary, visible_binary):
    occluded_gt = np.logical_and(target_binary == 1, visible_binary == 0).astype(np.uint8)
    has_occlusion = np.sum(occluded_gt) > 0

    intersection = np.logical_and(pred_binary == 1, target_binary == 1).sum()
    union = np.logical_or(pred_binary == 1, target_binary == 1).sum()
    iou = intersection / union if union > 0 else (1.0 if np.sum(target_binary) == 0 else 0.0)

    pred_sum = np.sum(pred_binary)
    target_sum = np.sum(target_binary)
    dice = (2.0 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else 1.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / target_sum if target_sum > 0 else 0.0

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
    print(f"🔍 Đánh giá Row 6 D2SA trên thiết bị: {device}")

    transform = A.Compose([A.Resize(args.resize, args.resize)])
    dataset = AmodalDatasetD2SA(img_dir=args.img_dir, ann_file=args.ann_file, transform=transform)

    if args.subset_size and args.subset_size < len(dataset):
        dataset.annotations = dataset.annotations[:args.subset_size]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = AmodalSwinUNet(num_classes=args.num_classes).to(device)
    print(f"📦 Đang nạp checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()

    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "inv_iou": 0.0}
    occluded_count = 0
    per_sample_metrics = []

    with torch.no_grad():
        for idx, (inputs, targets, _, class_ids) in enumerate(tqdm(loader, desc="Evaluating Row 6")):
            inputs = inputs.to(device)
            class_ids = class_ids.to(device)
            targets_np = targets.squeeze().numpy().astype(np.uint8)
            visible_np = inputs[:, 3, :, :].cpu().numpy().squeeze().astype(np.uint8)

            outputs = model(inputs, class_ids)
            preds_binary = (torch.sigmoid(outputs) > args.threshold).cpu().numpy().squeeze().astype(np.uint8)

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
    m_prec = (total_metrics["precision"] / n_samples) * 100
    m_rec = (total_metrics["recall"] / n_samples) * 100
    m_inv = (total_metrics["inv_iou"] / occluded_count * 100) if occluded_count > 0 else 0.0

    print(f"\n📊 KẾT QUẢ ROW 6: mIoU={m_iou:.2f}% | Dice={m_dice:.2f}% | Inv_mIoU={m_inv:.2f}%")

    results = {
        "config": "Row 6 (5ch + Emb 60 + Spatial Attention, OccLoss)",
        "summary_metrics": {
            "mIoU": float(m_iou), "dice": float(m_dice), "precision": float(m_prec),
            "recall": float(m_rec), "invisible_mIoU": float(m_inv)
        },
        "per_sample_metrics": per_sample_metrics
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Đã lưu tại: {args.output}")

    return results


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Đánh giá Swin-UNet Row 6 trên D2SA")
    parser.add_argument("--img-dir", type=str, default=os.path.join(root_dir, "data", "D2SA", "images"))
    parser.add_argument("--ann-file", type=str, default=os.path.join(root_dir, "data", "D2SA", "D2S_amodal_validation.json"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(root_dir, "checkpoints", "d2sa", "amodal_shape_prediction_full_config", "swin_amodal_epoch_30.pth")
    )
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=os.path.join(root_dir, "results", "d2sa", "row6_eval.json"))
    parser.add_argument("--subset-size", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
