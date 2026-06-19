"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL PREDICTION (FULL CONFIG: SPATIAL + CATEGORY)
===================================================================================
Script đánh giá hiệu suất mô hình trên từng ảnh (Instance-level).

Đã cập nhật:
- Fix lỗi "Halo Effect": Đồng bộ logic tính toán Occlusion GT bằng phép trừ tensor.
- Tối ưu hóa: Batch-size = 1, trích xuất ID nhãn cho mô hình Full Config.
- Độ chính xác: Loại bỏ nhiễu ranh giới do nội suy Resize.
===================================================================================
python scripts/evaluate.py --img-dir data/val2014 --ann-file data/annotations/COCO_amodal_val2014.json --checkpoint checkpoints/swin_amodal_epoch_30.pth
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
    Tính toán các metrics trực tiếp cho 1 ảnh (Instance-level).
    """
    # Ép kiểu và áp ngưỡng nhị phân cho các tensor đầu vào
    pred = (torch.sigmoid(pred_logits[0]) > threshold).float()
    target = (target[0] > 0.5).float()
    visible = (visible[0] > 0.5).float()

    # ─────────────────────────────────────────────────────────────
    # 1. Tính Overall mIoU & Dice (Toàn bộ vật thể)
    # ─────────────────────────────────────────────────────────────
    intersection = (pred * target).sum().item()
    union = (pred + target).clamp(0, 1).sum().item()
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    dice = (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)

    # 2. Precision & Recall (Pixel-level)
    precision = (intersection + 1e-7) / (pred_sum + 1e-7)
    recall = (intersection + 1e-7) / (target_sum + 1e-7)

    # ─────────────────────────────────────────────────────────────
    # 3. Tính Invisible IoU (Vùng che khuất - ĐÃ FIX LOGIC)
    # ─────────────────────────────────────────────────────────────
    # Sử dụng phép trừ để loại bỏ nhiễu nội suy ở ranh giới
    invisible_mask_gt = ((target - visible) > 0.5).float()
    pred_in_inv = ((pred - visible) > 0.5).float()
    
    inv_inter = (pred_in_inv * invisible_mask_gt).sum().item()
    inv_union = (pred_in_inv + invisible_mask_gt).clamp(0, 1).sum().item()
    
    inv_iou = (inv_inter + 1e-7) / (inv_union + 1e-7)
    has_occlusion = invisible_mask_gt.sum().item() > 0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "invisible_iou": inv_iou if has_occlusion else 0.0,
        "has_occlusion": has_occlusion
    }


def build_transform(resize):
    """Resize input về kích thước chuẩn của mô hình (224x224)."""
    return A.Compose([A.Resize(resize, resize)])


def evaluate(args):
    """Hàm điều phối quá trình đánh giá."""
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🔍 Đang đánh giá: Full Config | Thiết bị: {device}")

    # ─────────────────────────────────────────────────────────────
    # CHUẨN BỊ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────
    transform = build_transform(args.resize)
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    
    # Batch size = 1 để đảm bảo tính Instance-level thuần túy
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ─────────────────────────────────────────────────────────────
    # NẠP MÔ HÌNH (Yêu cầu khớp Class ID cho Category Embedding)
    # ─────────────────────────────────────────────────────────────
    model = AmodalSwinUNet(num_classes=91).to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Lỗi: Không tìm thấy tạ tại {args.checkpoint}")
        return
        
    # Xử lý bóc tách tiền tố _orig_mod từ torch.compile
    ckpt = torch.load(args.checkpoint, map_location=device)
    raw_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in raw_state_dict.items()}
    
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # ─────────────────────────────────────────────────────────────
    # VÒNG LẶP ĐÁNH GIÁ (SINGLE-IMAGE LEVEL)
    # ─────────────────────────────────────────────────────────────
    all_iou, all_dice, all_precision, all_recall = [], [], [], []
    all_inv_iou = []
    per_sample_metrics = []

    print("📊 Đang quét tập dữ liệu... Lần này điểm sẽ chuẩn hơn!")
    with torch.no_grad():
        for img_idx, (inputs, targets, _, class_ids) in enumerate(tqdm(loader, desc="Evaluating")):
            
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(device, non_blocking=True)
            visible_masks = inputs[:, 3:4, :, :].float().to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True) 

            # Suy luận tăng tốc với AMP
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, class_ids)
            
            # Tính toán metrics chi tiết cho từng ảnh
            metrics = calculate_single_image_metrics(outputs, targets, visible_masks, threshold=args.threshold)
            metrics["image_index"] = img_idx

            # Tổng hợp số liệu
            all_iou.append(metrics["iou"])
            all_dice.append(metrics["dice"])
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])
            
            if metrics["has_occlusion"]:
                all_inv_iou.append(metrics["invisible_iou"])
                
            per_sample_metrics.append(metrics)

    # ─────────────────────────────────────────────────────────────
    # XUẤT BÁO CÁO KẾT QUẢ
    # ─────────────────────────────────────────────────────────────
    n_samples = len(all_iou)
    m_iou = np.mean(all_iou) * 100 if all_iou else 0.0
    m_dice = np.mean(all_dice) * 100 if all_dice else 0.0
    m_precision = np.mean(all_precision) * 100 if all_precision else 0.0
    m_recall = np.mean(all_recall) * 100 if all_recall else 0.0
    m_inv_iou = np.mean(all_inv_iou) * 100 if all_inv_iou else 0.0

    print("\n" + "=" * 60)
    print(f"{'🏆 KẾT QUẢ ĐÁNH GIÁ (INSTANCE-LEVEL)':^60}")
    print("=" * 60)
    print(f"🎯 Overall mIoU      : {m_iou:.2f}%")
    print(f"🎲 Dice Coefficient  : {m_dice:.2f}%")
    print(f"✨ Mean Precision    : {m_precision:.2f}%")
    print(f"🔄 Mean Recall       : {m_recall:.2f}%")
    print(f"👁️  Invisible mIoU   : {m_inv_iou:.2f}%")
    print("-" * 60)
    print(f"📊 Tổng số mẫu       : {n_samples}")
    print("=" * 60)

    # Lưu JSON để Sếp vẽ biểu đồ phân tích lỗi
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results = {
            "overall_mIoU": float(m_iou),
            "invisible_mIoU": float(m_inv_iou),
            "samples": n_samples,
            "per_sample": per_sample_metrics
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Eval Amodal Swin-UNet")
    parser.add_argument("--img-dir", type=str, default="../data/val2014")
    parser.add_argument("--ann-file", type=str, default="../data/annotations/COCO_amodal_val2014.json")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/swin_amodal_epoch_30.pth")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results/full_config_eval.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)