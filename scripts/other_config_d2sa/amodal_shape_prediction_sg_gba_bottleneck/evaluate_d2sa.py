"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL SWIN-UNET (ROW 7 D2SA — PROPOSED: SG-GBA)
===================================================================================
Cấu hình: Row 7 Ablation Study:
- Đầu vào: 5 kênh (RGB + Visible Mask + Edge Mask)
- Category Embedding: Bật (60 lớp D2SA)
- Bottleneck Attention: SGGatedBottleneck (FiLM + Chú ý 7x7 + Learnable Gate)
- Spatial Attention ở cuối Decoder: Tắt

Chạy: python scripts/other_config_d2sa/amodal_shape_prediction_sg_gba_bottleneck/evaluate_d2sa.py
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
    print(f"🔍 Đánh giá Row 7 D2SA (SG-GBA) trên thiết bị: {device}")

    transform = A.Compose([A.Resize(args.resize, args.resize)])
    dataset = AmodalDatasetD2SA(img_dir=args.img_dir, ann_file=args.ann_file, transform=transform)

    if args.subset_size and args.subset_size < len(dataset):
        dataset.annotations = dataset.annotations[:args.subset_size]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    EXPECTED_PARAMS = 34_843_758
    model = AmodalSwinUNet(num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_PARAMS, (
        f"Lệch tham số: {total_params:,} != {EXPECTED_PARAMS:,}"
    )

    print(f"📦 Đang nạp checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()

    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "inv_iou": 0.0}
    occluded_count = 0
    per_sample_metrics = []

    with torch.no_grad():
        for idx, (inputs, targets, _, class_ids) in enumerate(tqdm(loader, desc="Evaluating Row 7")):
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
    m_iou = (total_metrics["iou"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_dice = (total_metrics["dice"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_prec = (total_metrics["precision"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_rec = (total_metrics["recall"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_inv = (total_metrics["inv_iou"] / occluded_count * 100) if occluded_count > 0 else 0.0

    print(f"\n📊 KẾT QUẢ ROW 7 (SG-GBA): mIoU={m_iou:.2f}% | Dice={m_dice:.2f}% | Inv_mIoU={m_inv:.2f}%")
    if hasattr(model, "sg_bottleneck") and hasattr(model.sg_bottleneck, "alpha"):
        print(f"   Learnable Gate Alpha: {float(model.sg_bottleneck.alpha.item()):.5f}")

    results = {
        "config": "Row 7 (5ch + Emb 60 + SG-GBA, OccLoss 5x)",
        "samples_evaluated": n_samples,
        "mIoU": round(m_iou, 2),
        "Dice": round(m_dice, 2),
        "Precision": round(m_prec, 2),
        "Recall": round(m_rec, 2),
        "Invisible_mIoU": round(m_inv, 2),
        "gate_alpha": round(float(model.sg_bottleneck.alpha.item()), 6),
        "details": per_sample_metrics,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"💾 Kết quả đã lưu tại: {args.output}")

    return results


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Đánh giá mô hình Row 7 D2SA (SG-GBA)")
    parser.add_argument("--img-dir", type=str, default=os.path.join(root_dir, "data", "D2SA", "images"), help="Thư mục ảnh")
    parser.add_argument("--ann-file", type=str, default=os.path.join(root_dir, "data", "D2SA", "D2S_amodal_validation.json"), help="File annotation validation")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(root_dir, "checkpoints", "d2sa", "row7_sg_gba_bottleneck", "swin_amodal_epoch_30.pth"), help="Checkpoint cần evaluate")
    parser.add_argument("--num-classes", type=int, default=60, help="Số lớp đối tượng")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng nhị phân hóa")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu test (0: toàn bộ)")
    parser.add_argument("--num-workers", type=int, default=2, help="Số worker nạp dữ liệu")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--output", type=str, default=os.path.join(root_dir, "results", "d2sa_row7_sg_gba_eval.json"), help="File lưu kết quả JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
