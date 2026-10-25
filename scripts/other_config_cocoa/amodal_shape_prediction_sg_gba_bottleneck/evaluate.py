"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH AMODAL SWIN-UNET (ROW 7 COCOA — PROPOSED: SG-GBA)
===================================================================================
Cấu hình: Row 7 Ablation Study trên COCO-Amodal:
- Đầu vào: 5 kênh (RGB + Visible Mask + Edge Mask)
- Category Embedding: Bật (91 lớp COCO)
- Bottleneck Attention: SGGatedBottleneck (FiLM + Chú ý 7x7 + Learnable Gate)
- Spatial Attention ở cuối Decoder: Tắt

Chạy: python scripts/other_config_cocoa/amodal_shape_prediction_sg_gba_bottleneck/evaluate.py
===================================================================================
"""

import argparse
import json
import os
import sys
import numpy as np

import albumentations as A
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from model import AmodalSwinUNet
from dataset import AmodalDataset


def calculate_single_image_metrics(pred_logits, target, visible, threshold=0.5):
    """Tính toán các metrics trực tiếp cho 1 ảnh (Instance-level)."""
    pred = (torch.sigmoid(pred_logits[0]) > threshold).float()
    target = (target[0] > 0.5).float()
    visible = (visible[0] > 0.5).float()

    intersection = (pred * target).sum().item()
    union = (pred + target).clamp(0, 1).sum().item()
    iou = (intersection + 1e-7) / (union + 1e-7)

    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    dice = (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)

    precision = (intersection + 1e-7) / (pred_sum + 1e-7)
    recall = (intersection + 1e-7) / (target_sum + 1e-7)

    invisible_mask_gt = ((target - visible) > 0.5).float()
    pred_in_inv = ((pred - visible) > 0.5).float()

    inv_inter = (pred_in_inv * invisible_mask_gt).sum().item()
    inv_union = (pred_in_inv + invisible_mask_gt).clamp(0, 1).sum().item()
    invisible_iou = (inv_inter + 1e-7) / (inv_union + 1e-7)
    has_occlusion = invisible_mask_gt.sum().item() > 0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "invisible_iou": float(invisible_iou) if has_occlusion else 0.0,
        "has_occlusion": bool(has_occlusion),
    }


def build_transform(resize):
    return A.Compose([A.Resize(resize, resize)])


def evaluate(args):
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🚀 Đang dùng thiết bị: {device}")

    transform = build_transform(args.resize)
    dataset = AmodalDataset(img_dir=args.img_dir, ann_file=args.ann_file, transform=transform)

    if args.subset_size and args.subset_size < len(dataset):
        dataset.annotations = dataset.annotations[:args.subset_size]

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    EXPECTED_PARAMS = 34_867_566
    model = AmodalSwinUNet(num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_PARAMS, f"Lệch tham số: {total_params} != {EXPECTED_PARAMS}"

    if not os.path.exists(args.checkpoint):
        print(f"❌ Lỗi: Không tìm thấy checkpoint tại {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device)
    raw_state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}

    model.load_state_dict(cleaned_state_dict)
    model.eval()

    all_iou, all_dice, all_precision, all_recall = [], [], [], []
    all_inv_iou = []
    per_sample_metrics = []

    print("📊 Đang quét tập dữ liệu COCOA Row 7...")
    with torch.no_grad():
        for img_idx, (inputs, targets, _, class_ids) in enumerate(tqdm(loader, desc="Evaluating Row 7")):
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            class_ids = class_ids.to(device)

            outputs = model(inputs, class_ids)

            metrics = calculate_single_image_metrics(
                pred_logits=outputs,
                target=targets,
                visible=inputs[:, 3:4, :, :],
                threshold=args.threshold
            )

            all_iou.append(metrics["iou"])
            all_dice.append(metrics["dice"])
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])

            if metrics["has_occlusion"]:
                all_inv_iou.append(metrics["invisible_iou"])

            per_sample_metrics.append({
                "image_index": img_idx,
                "class_id": int(class_ids.item()),
                "has_occlusion": metrics["has_occlusion"],
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "invisible_iou": metrics["invisible_iou"]
            })

    final_m_iou = float(np.mean(all_iou)) * 100 if all_iou else 0.0
    final_m_dice = float(np.mean(all_dice)) * 100 if all_dice else 0.0
    final_m_precision = float(np.mean(all_precision)) * 100 if all_precision else 0.0
    final_m_recall = float(np.mean(all_recall)) * 100 if all_recall else 0.0
    final_m_inv_iou = float(np.mean(all_inv_iou)) * 100 if all_inv_iou else 0.0

    print("=" * 65)
    print("📊 KẾT QUẢ ĐÁNH GIÁ ROW 7 (SG-GBA - COCOA)")
    print(f"Tổng số đối tượng kiểm thử : {len(all_iou)}")
    print(f"Số đối tượng có che khuất  : {len(all_inv_iou)}")
    print(f"Mean IoU (Overall)         : {final_m_iou:.2f}%")
    print(f"Mean Dice                  : {final_m_dice:.2f}%")
    print(f"Mean Precision             : {final_m_precision:.2f}%")
    print(f"Mean Recall                : {final_m_recall:.2f}%")
    print(f"Invisible mIoU (Vùng che)  : {final_m_inv_iou:.2f}%")
    print("=" * 65)

    report_data = {
        "config": "Row 7 (COCOA: 5ch + Emb 91 + SG-GBA, OccLoss 5x)",
        "summary": {
            "total_images": len(all_iou),
            "occluded_images": len(all_inv_iou),
            "mIoU": round(final_m_iou, 2),
            "Dice": round(final_m_dice, 2),
            "Precision": round(final_m_precision, 2),
            "Recall": round(final_m_recall, 2),
            "Invisible_mIoU": round(final_m_inv_iou, 2),
            "gate_alpha": round(float(model.sg_bottleneck.alpha.item()), 6),
        },
        "per_sample": per_sample_metrics
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print(f"💾 File JSON lưu tại: {args.output}")

    return report_data


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Đánh giá mô hình Row 7 COCOA (SG-GBA)")
    parser.add_argument("--img-dir", type=str, default=os.path.join(root_dir, "data", "val2014"), help="Thư mục ảnh")
    parser.add_argument("--ann-file", type=str, default=os.path.join(root_dir, "data", "annotations", "COCO_amodal_val2014.json"), help="File annotation validation")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(root_dir, "checkpoints", "cocoa", "row7_sg_gba_bottleneck", "swin_amodal_epoch_30.pth"), help="Checkpoint cần evaluate")
    parser.add_argument("--num-classes", type=int, default=91, help="Số lớp đối tượng (91)")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng nhị phân hóa")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu test (0: toàn bộ)")
    parser.add_argument("--num-workers", type=int, default=2, help="Số worker nạp dữ liệu")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--output", type=str, default=os.path.join(root_dir, "results", "cocoa_row7_sg_gba_eval.json"), help="File lưu kết quả JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
