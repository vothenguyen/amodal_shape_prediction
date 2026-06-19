"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL PREDICTION (SINGLE-IMAGE EVALUATION)
===================================================================================
Script đánh giá hiệu suất mô hình trên validation set, xử lý từng ảnh một.

Cập nhật:
- Batch size mặc định = 1 để hỗ trợ phân tích chi tiết từng vật thể.
- Tính toán Precision, Recall và Invisible IoU độc lập cho mỗi mẫu.

Chạy: python scripts/evaluate.py --img-dir data/val2014 --ann-file data/annotations/COCO_amodal_val2014.json --checkpoint checkpoints/swin_amodal_epoch_30.pth
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


def calculate_single_metrics(pred_logits, target, visible, threshold=0.5):
    """
    Tính toán metrics cho 1 ảnh duy nhất (Shape: [1, 1, H, W]).
    """
    # Loại bỏ chiều batch [0] để tính toán trên mặt phẳng 2D
    pred = (torch.sigmoid(pred_logits[0, 0]) > threshold).float()
    gt = (target[0, 0] > 0.5).float()
    vis = (visible[0, 0] > 0.5).float()

    # ─────────────────────────────────────────────────────────────
    # 1. Overall Metrics (IoU, Dice, Precision, Recall)
    # ─────────────────────────────────────────────────────────────
    intersection = (pred * gt).sum().item()
    union = (pred + gt).clamp(0, 1).sum().item()
    
    p_sum = pred.sum().item()
    t_sum = gt.sum().item()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (p_sum + t_sum + 1e-7)
    precision = (intersection + 1e-7) / (p_sum + 1e-7)
    recall = (intersection + 1e-7) / (t_sum + 1e-7)

    # ─────────────────────────────────────────────────────────────
    # 2. Invisible IoU (Occlusion Region) - ĐÃ FIX LOGIC CHUẨN
    # ─────────────────────────────────────────────────────────────
    invisible_target = ((gt - vis) > 0.5).float()
    pred_invisible = ((pred - vis) > 0.5).float()

    inv_inter = (pred_invisible * invisible_target).sum().item()
    inv_union = (pred_invisible + invisible_target).clamp(0, 1).sum().item()
    
    has_occlusion = invisible_target.sum().item() > 0
    inv_iou = (inv_inter + 1e-7) / (inv_union + 1e-7) if has_occlusion else 0.0

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
    print(f"🔍 Đánh giá trên thiết bị: {device} | Mode: Single-Image")

    # CHUẨN BỊ DỮ LIỆU
    transform = build_transform(args.resize)
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    # Khóa batch_size = 1
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # NẠP MÔ HÌNH
    model = AmodalSwinUNet().to(device)
    
    # Hỗ trợ bóc tách checkpoint từ torch.compile hoặc model_state_dict
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()

    all_metrics = []
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "inv_iou": 0}
    occluded_count = 0

    print("📊 Đang chấm điểm từng ảnh...")
    with torch.no_grad():
        for idx, (inputs, targets, _, _) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            visible_masks = inputs[:, 3:4, :, :].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
            
            res = calculate_single_metrics(outputs, targets, visible_masks, threshold=args.threshold)
            res["image_index"] = idx
            
            # Cộng dồn để tính mIoU cuối cùng
            total_metrics["iou"] += res["iou"]
            total_metrics["dice"] += res["dice"]
            total_metrics["precision"] += res["precision"]
            total_metrics["recall"] += res["recall"]
            
            if res["has_occlusion"]:
                total_metrics["inv_iou"] += res["invisible_iou"]
                occluded_count += 1
                
            all_metrics.append(res)

    # TÍNH TRUNG BÌNH
    n = len(dataset)
    m_iou = (total_metrics["iou"] / n) * 100
    m_dice = (total_metrics["dice"] / n) * 100
    m_prec = (total_metrics["precision"] / n) * 100
    m_rec = (total_metrics["recall"] / n) * 100
    m_inv_iou = (total_metrics["inv_iou"] / occluded_count * 100) if occluded_count > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"{'🏆 KẾT QUẢ ĐÁNH GIÁ CHI TIẾT':^60}")
    print("=" * 60)
    print(f"📊 Tổng số mẫu      : {n}")
    print(f"🎯 Overall mIoU     : {m_iou:.2f}%")
    print(f"🎲 Dice Coefficient  : {m_dice:.2f}%")
    print(f"✨ Mean Precision    : {m_prec:.2f}%")
    print(f"🔄 Mean Recall       : {m_rec:.2f}%")
    print(f"👁️  Invisible mIoU   : {m_inv_iou:.2f}%")
    print("=" * 60)

    if args.output:
        output_data = {
            "summary": {
                "mIoU": m_iou, "dice": m_dice, "precision": m_prec, 
                "recall": m_rec, "invisible_mIoU": m_inv_iou
            },
            "per_image": all_metrics
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Đã lưu log chi tiết tại: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--ann-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results/eval_log.json")
    args = parser.parse_args()
    evaluate(args)