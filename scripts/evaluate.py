"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL PREDICTION (Cập nhật Precision & Recall - SINGLE IMAGE)
===================================================================================
Script đánh giá hiệu suất mô hình trên validation set.

Tối ưu hóa:
- Chế độ Single-Image: Khóa batch_size = 1, tính toán ma trận phẳng.
- Bắt trực tiếp Index của từng bức ảnh để phục vụ Error Analysis.
- Tự động bóc tách tạ (Hỗ trợ checkpoint từ torch.compile).

Metrics used:
- IoU (Intersection over Union): Tính lên toàn bộ mask amodal
- Dice Coefficient: F1-score cho segmentation
- Precision: Độ chính xác của các pixel được dự đoán (Pixel-level)
- Recall: Độ phủ của các pixel được dự đoán so với thực tế (Pixel-level)
- Invisible IoU: IoU chỉ tính trên vùng bị che khuất (occlusion region)

Chạy: python scripts/evaluate_2.py --img-dir data/val2014 --ann-file data/annotations/COCO_amodal_val2014.json --checkpoint checkpoints/our_base_cocoa_epoch_30.pth
===================================================================================
"""

import argparse
import json
import os
import numpy as np

import albumentations as A
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AmodalSwinUNet
from dataset import AmodalDataset


def calculate_single_image_metrics(pred_logits, target, visible, threshold=0.5):
    """
    Tính toán metrics trực tiếp cho một ảnh duy nhất (Input shape: [1, C, H, W]).
    """
    # Ép kiểu và áp dụng threshold (Loại bỏ chiều batch index 0)
    p = (torch.sigmoid(pred_logits[0]) > threshold).float()
    t = (target[0] > 0.5).float()
    v = (visible[0] > 0.5).float()

    # ─────────────────────────────────────────────────────────────
    # 1. Tính Overall Metrics (IoU, Dice, Precision, Recall)
    # ─────────────────────────────────────────────────────────────
    inter_o = (p * t).sum().item()
    union_o = (p + t).clamp(0, 1).sum().item()
    
    p_sum = p.sum().item()
    t_sum = t.sum().item()

    iou = inter_o / union_o if union_o > 0 else 1.0
    dice = (2.0 * inter_o) / (p_sum + t_sum) if (p_sum + t_sum) > 0 else 1.0
    
    # Precision = TP / (TP + FP)
    precision = inter_o / p_sum if p_sum > 0 else 1.0
    # Recall = TP / (TP + FN)
    recall = inter_o / t_sum if t_sum > 0 else 1.0

    # ─────────────────────────────────────────────────────────────
    # 2. Tính Invisible IoU (Vùng bị che khuất) - ĐÃ CHUẨN HÓA LOGIC
    # ─────────────────────────────────────────────────────────────
    invisible_target = ((t - v) > 0.5).float()
    pred_invisible = ((p - v) > 0.5).float()
    
    inv_inter = (pred_invisible * invisible_target).sum().item()
    inv_union = (pred_invisible + invisible_target).clamp(0, 1).sum().item()
    inv_t_sum = invisible_target.sum().item()

    has_occlusion = inv_t_sum > 0
    inv_iou = inv_inter / inv_union if inv_union > 0 else -1.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "invisible_iou": inv_iou,
        "has_occlusion": has_occlusion
    }


def build_transform(resize):
    return A.Compose([A.Resize(resize, resize)])


def evaluate(args):
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🔍 Đánh giá trên thiết bị: {device} | Chế độ: Từng ảnh (Single-Image)")

    # CHUẨN BỊ DỮ LIỆU
    transform = build_transform(args.resize)
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    
    # Khóa cứng batch_size = 1
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ─────────────────────────────────────────────────────────────
    # NẠP MÔ HÌNH VÀ BÓC TÁCH CHECKPOINT (FIX LỖI _orig_mod)
    # ─────────────────────────────────────────────────────────────
    model = AmodalSwinUNet(num_classes=91).to(device)
    
    # Nạp file tạ
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Rút phần lõi (nếu là dạng dictionary của quá trình train)
    raw_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Lột bỏ cái mác '_orig_mod.' do torch.compile sinh ra
    cleaned_state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in raw_state_dict.items()}
    
    # Bơm tạ sạch vào model
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # KHỞI TẠO BIẾN LƯU TRỮ
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "inv_iou": 0}
    occluded_count = 0
    per_sample_metrics = []

    print("📊 Bắt đầu tính toán metrics trên từng ảnh...")
    with torch.no_grad():
        # Dùng enumerate để lấy index (mô phỏng ID của ảnh)
        for img_idx, (inputs, targets, occluded_region, class_ids) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            visible_masks = inputs[:, 3:4, :, :].float().to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs, class_ids)
            
            # Tính metric trực tiếp cho 1 ảnh
            metrics = calculate_single_image_metrics(outputs, targets, visible_masks, threshold=args.threshold)
            
            # Gắn thêm ID/Index ảnh để dễ phân tích
            metrics["image_index"] = img_idx

            # Tổng hợp kết quả
            total_metrics["iou"] += metrics["iou"]
            total_metrics["dice"] += metrics["dice"]
            total_metrics["precision"] += metrics["precision"]
            total_metrics["recall"] += metrics["recall"]
            
            if metrics["has_occlusion"]:
                total_metrics["inv_iou"] += metrics["invisible_iou"]
                occluded_count += 1
                
            per_sample_metrics.append(metrics)

    # TÍNH TRUNG BÌNH & IN KẾT QUẢ
    n_samples = len(dataset)
    m_iou = (total_metrics["iou"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_dice = (total_metrics["dice"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_precision = (total_metrics["precision"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_recall = (total_metrics["recall"] / n_samples) * 100 if n_samples > 0 else 0.0
    m_inv_iou = (total_metrics["inv_iou"] / occluded_count) * 100 if occluded_count > 0 else 0.0

    print("\n" + "=" * 60)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CHI TIẾT (SINGLE-IMAGE LEVEL) 📘")
    print("=" * 60)
    print(f"📂 Dataset           : {args.ann_file}")
    print(f"📦 Checkpoint        : {args.checkpoint}")
    print(f"📊 Tổng số mẫu       : {n_samples}")
    print(f"🎯 Overall mIoU      : {m_iou:.2f}%")
    print(f"🎲 Dice Coefficient  : {m_dice:.2f}%")
    print(f"✨ Mean Precision    : {m_precision:.2f}%")
    print(f"🔄 Mean Recall       : {m_recall:.2f}%")
    print(f"👁️  Invisible mIoU   : {m_inv_iou:.2f}%")
    print("=" * 60)

    # LƯU KẾT QUẢ
    results = {
        "dataset": args.ann_file,
        "checkpoint": args.checkpoint,
        "total_images": n_samples,
        "summary_metrics": {
            "overall_mIoU": float(m_iou),
            "dice": float(m_dice),
            "precision": float(m_precision),
            "recall": float(m_recall),
            "invisible_mIoU": float(m_inv_iou)
        },
        "settings": {
            "threshold": args.threshold,
            "resize": args.resize,
            "device": str(device)
        },
        "per_sample_metrics": per_sample_metrics,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Kết quả chi tiết từng ảnh lưu tại: {args.output}")

    return results


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="Đánh giá mô hình Amodal Segmentation (Từng ảnh)")
    parser.add_argument("--img-dir", type=str, default=os.path.join(root_dir, "data", "val2014"), help="Thư mục chứa ảnh validation")
    parser.add_argument("--ann-file", type=str, default=os.path.join(root_dir, "data", "annotations", "COCO_amodal_val2014.json"), help="File annotation")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(root_dir, "checkpoints", "swin_amodal_epoch_30.pth"), help="Đường dẫn checkpoint")
    
    # Loại bỏ --batch-size vì đã fix cứng là 1 trong code
    parser.add_argument("--num-workers", type=int, default=4, help="Số worker DataLoader")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize input")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng sigmoid")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cpu, cuda")
    parser.add_argument("--output", type=str, default=os.path.join(root_dir, "results", "per_image_eval.json"), help="File lưu kết quả")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)