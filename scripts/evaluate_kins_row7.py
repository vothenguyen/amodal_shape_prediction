"""
===================================================================================
ĐÁNH GIÁ MÔ HÌNH ROW 7 (SG-GBA SWIN-UNET) TRÊN KINS
===================================================================================
Tính toán:
- Overall mIoU (Mean IoU trên tất cả đối tượng)
- Occlusion mIoU (Chỉ tính trên đối tượng có phần bị che khuất: area > 0)
- Pixel-wise Precision (mAP)
- Pixel-wise Recall (mAR)
- Xuất báo cáo JSON vào results/
===================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset_kins import KINSDataset
from model_kins_row7 import AmodalSwinUNetRow7KINS


def evaluate(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🚀 Bắt đầu đánh giá Row 7 KINS trên thiết bị: {device}")

    transform = A.Compose([A.Resize(args.resize, args.resize)])
    dataset = KINSDataset(img_dir=args.img_dir, ann_file=args.ann_file, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    EXPECTED_PARAMS = 34_813_038
    model = AmodalSwinUNetRow7KINS(pretrained=False, num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_PARAMS, f"Lệch tham số: {total_params} != {EXPECTED_PARAMS}"

    if not os.path.exists(args.checkpoint):
        print(f"❌ Không tìm thấy checkpoint tại: {args.checkpoint}")
        return None

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state)
    model.eval()

    ious = []
    occ_ious = []
    precisions = []
    recalls = []

    print(f"🔍 Đang đánh giá {len(dataset):,} mẫu kiểm thử KINS...")
    with torch.no_grad():
        for inputs, targets, occluded, class_ids in tqdm(loader, desc="Evaluating Row 7 KINS"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(device, non_blocking=True)
            occluded = occluded.unsqueeze(1).float().to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs, class_ids)
                pred_masks = (torch.sigmoid(outputs) > args.threshold).float()

            TP = (pred_masks * targets).sum(dim=(2, 3))
            FP = (pred_masks * (1 - targets)).sum(dim=(2, 3))
            FN = ((1 - pred_masks) * targets).sum(dim=(2, 3))

            batch_iou = (TP + 1e-6) / (TP + FP + FN + 1e-6)
            batch_precision = (TP + 1e-6) / (TP + FP + 1e-6)
            batch_recall = (TP + 1e-6) / (TP + FN + 1e-6)

            ious.extend(batch_iou.cpu().numpy().flatten().tolist())
            precisions.extend(batch_precision.cpu().numpy().flatten().tolist())
            recalls.extend(batch_recall.cpu().numpy().flatten().tolist())

            has_occlusion = occluded.sum(dim=(2, 3)) > 0
            if has_occlusion.any():
                occ_intersection = (pred_masks * targets * occluded).sum(dim=(2, 3))
                occ_union = (pred_masks * occluded).sum(dim=(2, 3)) + (targets * occluded).sum(dim=(2, 3)) - occ_intersection
                occ_batch_iou = (occ_intersection + 1e-6) / (occ_union + 1e-6)
                valid_occ = occ_batch_iou[has_occlusion.squeeze(1)]
                occ_ious.extend(valid_occ.cpu().numpy().flatten().tolist())

    final_miou = float(np.mean(ious)) * 100 if ious else 0.0
    final_occ_miou = float(np.mean(occ_ious)) * 100 if occ_ious else 0.0
    final_precision = float(np.mean(precisions)) * 100 if precisions else 0.0
    final_recall = float(np.mean(recalls)) * 100 if recalls else 0.0
    gate_alpha = float(model.sg_bottleneck.alpha.item())

    print("\n" + "=" * 65)
    print("📊 KẾT QUẢ ĐÁNH GIÁ ROW 7 (SG-GBA - KINS)")
    print("=" * 65)
    print(f"Tổng số đối tượng kiểm thử : {len(ious):,}")
    print(f"Số đối tượng có che khuất  : {len(occ_ious):,}")
    print(f"Overall mIoU               : {final_miou:.2f}%")
    print(f"Occlusion mIoU             : {final_occ_miou:.2f}%")
    print(f"Precision (mAP)            : {final_precision:.2f}%")
    print(f"Recall (mAR)               : {final_recall:.2f}%")
    print(f"Learnable Gate Alpha       : {gate_alpha:.6f}")
    print("=" * 65)

    report_data = {
        "dataset": "KINS",
        "configuration": "Row 7 - SG-GBA (Proposed)",
        "summary": {
            "total_objects": len(ious),
            "occluded_objects": len(occ_ious),
            "mIoU": round(final_miou, 2),
            "Occlusion_mIoU": round(final_occ_miou, 2),
            "Precision": round(final_precision, 2),
            "Recall": round(final_recall, 2),
            "gate_alpha": round(gate_alpha, 6),
        }
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print(f"💾 File JSON lưu tại: {args.output}")

    return report_data


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Row 7 KINS (SG-GBA)")
    parser.add_argument("--img-dir", type=str, default="/content/kitti_data/testing/image_2", help="Thư mục ảnh test")
    parser.add_argument("--ann-file", type=str, default="/content/kins_data/update_test_2020.json", help="File annotation KINS test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/kins/row7_sg_gba/swin_amodal_kins_row7_epoch_30.pth", help="Checkpoint cần evaluate")
    parser.add_argument("--num-classes", type=int, default=20, help="Số lớp đối tượng (20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size test")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước resize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng nhị phân hóa")
    parser.add_argument("--num-workers", type=int, default=4, help="Số worker nạp dữ liệu")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--output", type=str, default="results/kins_row7_sg_gba_eval.json", help="File lưu JSON kết quả")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
