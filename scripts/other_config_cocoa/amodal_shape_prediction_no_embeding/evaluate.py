"""
===================================================================================
MÔ HÌNH AMODAL SWIN-UNET - Dự đoán hình dạng toàn bộ của vật thể che khuất
===================================================================================
Kiến trúc: Swin Transformer Encoder (5 kênh) + U-Net Decoder + Spatial Attention
- Nhập liệu: RGB (3) + Visible mask (1) + Edge mask (1)
- Đầu ra: Amodal mask (1)
- Chế độ đánh giá: SINGLE-IMAGE (Batch-size = 1, trích xuất ID từng ảnh)

Metrics used (Instance-level):
- Overall mIoU: Tính lên toàn bộ mask amodal
- Occlusion mIoU: IoU chỉ tính trên vùng bị che khuất
- Occlusion Recall: Độ phủ trên vùng che khuất

Chạy: python scripts/evaluate.py --img-dir data/val2014 --ann-file data/annotations/COCO_amodal_val2014.json --checkpoint checkpoints/swin_amodal_epoch_29.pth
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
    # Ép kiểu, áp threshold và loại bỏ chiều batch [0]
    pred = (torch.sigmoid(pred_logits[0]) > threshold).float()
    target = (target[0] > 0.5).float()
    visible = (visible[0] > 0.5).float()

    # Tính mask vùng bị che khuất (ĐÃ FIX CHUẨN LOGIC TOÁN HỌC)
    gt_occ = ((target - visible) > 0.5).float()
    pred_occ = ((pred - visible) > 0.5).float()

    # 1. Overall IoU (Amodal) - (ĐÃ FIX BẢO VỆ UNION)
    inter_o = (pred * target).sum()
    union_o = (pred + target).clamp(0, 1).sum()
    overall_iou = (inter_o / union_o).item() if union_o > 0 else 1.0

    # 2. Occlusion Metrics - (ĐÃ FIX BẢO VỆ UNION)
    inter_occ = (pred_occ * gt_occ).sum()
    union_occ = (pred_occ + gt_occ).clamp(0, 1).sum()
    gt_occ_sum = gt_occ.sum()

    has_occlusion = gt_occ_sum.item() > 0
    occ_iou = (inter_occ / union_occ).item() if union_occ > 0 else 0.0
    occ_recall = (inter_occ / gt_occ_sum).item() if has_occlusion else 0.0

    return {
        "iou": overall_iou,
        "occ_iou": occ_iou,
        "occ_recall": occ_recall,
        "has_occlusion": has_occlusion
    }


def build_transform(resize):
    """
    Xây dựng augmentation pipeline cho evaluation.
    Lưu ý: Evaluation không dùng augmentation, chỉ resize
    """
    return A.Compose([A.Resize(resize, resize)])


def evaluate(args):
    """
    Hàm chính để đánh giá mô hình.
    """
    # Chọn thiết bị
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🔍 Đánh giá trên thiết bị: {device} | Chế độ: Từng ảnh (Batch-size=1)")

    # ─────────────────────────────────────────────────────────────
    # CHUẨN BỊ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────
    transform = build_transform(args.resize)
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    
    # BẮT BUỘC batch_size=1 để chấm từng ảnh
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    # ─────────────────────────────────────────────────────────────
    # NẠP MÔ HÌNH (SPATIAL ATTENTION - KHÔNG CÓ CATEGORY EMBEDDING)
    # ─────────────────────────────────────────────────────────────
    model = AmodalSwinUNet().to(device)
    
    # Bóc tách tiền tố _orig_mod (Nếu mô hình được train với torch.compile)
    ckpt = torch.load(args.checkpoint, map_location=device)
    raw_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in raw_state_dict.items()}
    
    model.load_state_dict(cleaned_state_dict)
    model.eval()  # Bật chế độ evaluation

    # ─────────────────────────────────────────────────────────────
    # VÒNG LẶP ĐÁNH GIÁ (SINGLE-IMAGE LEVEL)
    # ─────────────────────────────────────────────────────────────
    all_overall = []
    all_occ_iou = []
    all_occ_recall = []
    per_sample_metrics = []

    print("📊 Tính toán metrics từng ảnh... Xin chờ!")
    with torch.no_grad():
        # Dùng enumerate để gán index cho từng ảnh
        for img_idx, (inputs, targets, occluded_region, _) in enumerate(tqdm(loader, desc="Evaluating")):
            # Di chuyển dữ liệu lên device
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            visible_masks = inputs[:, 3:4, :, :].float().to(device)  # Kênh 4 là visible mask

            # Suy luận
            outputs = model(inputs)
                
            # Tính toán chỉ số trực tiếp cho 1 ảnh
            metrics = calculate_single_image_metrics(outputs, targets, visible_masks, threshold=args.threshold)
            metrics["image_index"] = img_idx  # Gắn ID ảnh để trace lỗi sau này

            # Tổng hợp kết quả
            all_overall.append(metrics["iou"])
            if metrics["has_occlusion"]:
                all_occ_iou.append(metrics["occ_iou"])
                all_occ_recall.append(metrics["occ_recall"])
            
            per_sample_metrics.append(metrics)

    # ─────────────────────────────────────────────────────────────
    # TÍNH TRUNG BÌNH & IN KẾT QUẢ
    # ─────────────────────────────────────────────────────────────
    n_samples = len(all_overall)
    m_iou = np.mean(all_overall) * 100 if all_overall else 0.0
    m_occ_iou = np.mean(all_occ_iou) * 100 if all_occ_iou else 0.0
    m_occ_recall = np.mean(all_occ_recall) * 100 if all_occ_recall else 0.0

    print("\n" + "=" * 60)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ (SPATIAL + NO CATEGORY EMBEDDING) 📘")
    print("=" * 60)
    print(f"📂 Dataset           : {args.ann_file}")
    print(f"📦 Checkpoint        : {args.checkpoint}")
    print(f"📊 Tổng số ảnh       : {n_samples}")
    print(f"🎯 Overall mIoU      : {m_iou:.2f}%")
    print(f"👻 Occlusion mIoU    : {m_occ_iou:.2f}%")
    print(f"🔍 Occ Recall        : {m_occ_recall:.2f}%")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # LƯU KẾT QUẢ
    # ─────────────────────────────────────────────────────────────
    results = {
        "dataset": args.ann_file,
        "checkpoint": args.checkpoint,
        "total_images": n_samples,
        "overall_mIoU": float(m_iou),
        "occlusion_mIoU": float(m_occ_iou),
        "occlusion_recall": float(m_occ_recall),
        "threshold": args.threshold,
        "resize": args.resize,
        "device": str(device),
        "per_image_metrics": per_sample_metrics,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Kết quả lưu tại: {args.output}")

    return results


def parse_args():
    """Phân tích command-line arguments."""
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Amodal Segmentation (Từng ảnh)")
    parser.add_argument("--img-dir", type=str, default="../data/val2014", help="Thư mục chứa ảnh validation")
    parser.add_argument("--ann-file", type=str, default="../data/annotations/COCO_amodal_val2014.json", help="File annotation COCO-Amodal")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/swin_amodal_epoch_29.pth", help="Đường dẫn checkpoint mô hình")
    # Đã xóa --batch-size vì code đã khóa cứng thành 1
    parser.add_argument("--num-workers", type=int, default=4, help="Số worker DataLoader")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize input")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng sigmoid để tạo binary mask")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cpu, hoặc cuda")
    parser.add_argument("--output", type=str, default="results/per_image_metrics_spatial.json", help="Lưu kết quả ra file JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)