"""
===================================================================================
ĐÁNH GIÁ CHUẨN KINS THEO THƯỚC ĐO MAPR (INVISIBLE mIOU)
===================================================================================
Tự động nhận diện kiến trúc:
- Row 7: SG-GBA Swin-UNet (AmodalSwinUNetRow7KINS, 34,813,038 params)
- Row 4: Baseline Swin-UNet (AmodalSwinUNet, 34,368,923 params)

Thước đo chuẩn MAPR (Công thức 1):
- Overall mIoU (%): (pred & target).sum() / (pred | target).sum()
- Invisible mIoU (%) [Công thức 1 MAPR chuẩn]:
    pred_occ = clip(pred - visible, 0, 1)
    gt_occ   = clip(target - visible, 0, 1)
    inter_occ = (pred_occ & gt_occ).sum()
    union_occ = (pred_occ | gt_occ).sum()
    inv_iou   = inter_occ / union_occ (chỉ tính trên các mẫu có gt_occ > 0)
- Occlusion Recall (%) [Công thức 2 Coverage/Recall cũ để đối chiếu]:
    occ_recall = inter_occ / gt_occ.sum()
- Pixel-wise Precision (mAP) (%)
- Pixel-wise Recall (mAR) (%)
- Cổng alpha (đối với mô hình có SG-GBA Bottleneck)
===================================================================================
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows shell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.dataset_kins import KINSDataset
from scripts.model_kins_row7 import AmodalSwinUNetRow7KINS
from scripts.model import AmodalSwinUNet


def find_path(user_path, candidates):
    """Tìm đường dẫn hợp lệ từ danh sách các ứng viên phổ biến."""
    if user_path and os.path.exists(user_path):
        return user_path
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return user_path


def load_model(ckpt_path, num_classes=20, device="cpu"):
    """
    Nạp checkpoint và tự động nhận diện kiến trúc:
    - Nếu có 'sg_bottleneck': khởi tạo AmodalSwinUNetRow7KINS (Row 7 SG-GBA)
    - Nếu không: khởi tạo AmodalSwinUNet (Row 4 MAPR Baseline)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    # Kiểm tra sự hiện diện của khối SG-GBA
    is_row7 = any("sg_bottleneck" in k for k in cleaned_state.keys())

    if is_row7:
        model = AmodalSwinUNetRow7KINS(pretrained=False, num_classes=num_classes).to(device)
        model.load_state_dict(cleaned_state)
        model_type = "Row 7 (SG-GBA Bottleneck)"
        gate_alpha = float(model.sg_bottleneck.alpha.item())
    else:
        model = AmodalSwinUNet(num_classes=num_classes).to(device)
        model.load_state_dict(cleaned_state)
        model_type = "Row 4 (MAPR Baseline)"
        gate_alpha = None

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    return model, model_type, total_params, gate_alpha


def evaluate_single_checkpoint(model, loader, device, threshold=0.5, desc="Evaluating"):
    """
    Đánh giá một mô hình trên DataLoader theo đúng chuẩn MAPR.
    Tính toán trên GPU tensor theo batch để đạt tốc độ tối đa.
    """
    ious = []
    precisions = []
    recalls = []
    inv_ious = []
    occ_recalls = []

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
    with torch.no_grad():
        for inputs, targets, occluded, class_ids in pbar:
            # inputs: [B, 5, H, W]
            # targets: [B, H, W]
            # occluded: [B, H, W]
            # class_ids: [B]
            targets_gpu = targets.unsqueeze(1).float().to(device, non_blocking=True)
            vis_gpu = inputs[:, 3:4, :, :].to(device, non_blocking=True)
            inputs_gpu = inputs.to(device, non_blocking=True)
            class_ids_gpu = class_ids.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs_gpu, class_ids_gpu)
                preds = (torch.sigmoid(outputs) > threshold).float()

            # -----------------------------------------------------------
            # 1. OVERALL METRICS (trên toàn bộ amodal mask)
            # -----------------------------------------------------------
            tp = (preds * targets_gpu).sum(dim=(1, 2, 3))
            fp = (preds * (1.0 - targets_gpu)).sum(dim=(1, 2, 3))
            fn = ((1.0 - preds) * targets_gpu).sum(dim=(1, 2, 3))
            union = tp + fp + fn

            batch_iou = torch.where(union > 0, tp / union, torch.tensor(1.0, device=device))
            batch_prec = torch.where((tp + fp) > 0, tp / (tp + fp), torch.tensor(0.0, device=device))
            batch_rec = torch.where((tp + fn) > 0, tp / (tp + fn), torch.tensor(0.0, device=device))

            ious.extend(batch_iou.cpu().tolist())
            precisions.extend(batch_prec.cpu().tolist())
            recalls.extend(batch_rec.cpu().tolist())

            # -----------------------------------------------------------
            # 2. OCCLUSION METRICS (Công thức 1 Invisible mIoU & Công thức 2 Recall)
            # -----------------------------------------------------------
            gt_occ = torch.clamp(targets_gpu - vis_gpu, 0.0, 1.0)
            pred_occ = torch.clamp(preds - vis_gpu, 0.0, 1.0)

            inter_occ = (pred_occ * gt_occ).sum(dim=(1, 2, 3))
            union_occ = ((pred_occ + gt_occ) > 0).float().sum(dim=(1, 2, 3))
            gt_occ_sum = gt_occ.sum(dim=(1, 2, 3))

            has_occ = (gt_occ_sum > 0)

            # Công thức 1: Invisible mIoU (chuẩn MAPR)
            batch_inv_iou = torch.where(union_occ > 0, inter_occ / union_occ, torch.tensor(0.0, device=device))
            # Công thức 2: Occlusion Recall (độ phủ / coverage của phần che khuất)
            batch_occ_rec = torch.where(gt_occ_sum > 0, inter_occ / gt_occ_sum, torch.tensor(0.0, device=device))

            if has_occ.any():
                inv_ious.extend(batch_inv_iou[has_occ].cpu().tolist())
                occ_recalls.extend(batch_occ_rec[has_occ].cpu().tolist())

            # Cập nhật thông tin nhanh trên thanh pbar
            cur_overall = np.mean(ious) * 100 if ious else 0.0
            cur_inv = np.mean(inv_ious) * 100 if inv_ious else 0.0
            pbar.set_postfix({"Overall": f"{cur_overall:.2f}%", "Inv_mIoU": f"{cur_inv:.2f}%"})

    results = {
        "total_instances": len(ious),
        "occluded_instances": len(inv_ious),
        "overall_miou": float(np.mean(ious) * 100) if ious else 0.0,
        "invisible_miou": float(np.mean(inv_ious) * 100) if inv_ious else 0.0,
        "occlusion_recall": float(np.mean(occ_recalls) * 100) if occ_recalls else 0.0,
        "precision": float(np.mean(precisions) * 100) if precisions else 0.0,
        "recall": float(np.mean(recalls) * 100) if recalls else 0.0,
    }

    return results


def build_markdown_table(records):
    """Tạo bảng so sánh kết quả dạng Markdown chuyên nghiệp."""
    lines = [
        "| Checkpoint | Kiến trúc | Tham số | Cổng $\\alpha$ | Tổng mẫu | Mẫu che khuất | Overall mIoU (%) | Invisible mIoU (F1) (%) | Occlusion Recall (F2) (%) | Precision (%) | Recall (%) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        ckpt_name = os.path.basename(r["checkpoint"])
        gate_str = f"{r['gate_alpha']:.4f}" if r["gate_alpha"] is not None else "N/A"
        lines.append(
            f"| `{ckpt_name}` | {r['model_type']} | {r['total_params']:,} | {gate_str} | "
            f"{r['total_instances']:,} | {r['occluded_instances']:,} | "
            f"**{r['overall_miou']:.2f}%** | **{r['invisible_miou']:.2f}%** | "
            f"{r['occlusion_recall']:.2f}% | {r['precision']:.2f}% | {r['recall']:.2f}% |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá chuẩn KINS theo thước đo MAPR (Invisible mIoU & Occlusion Recall)"
    )
    parser.add_argument("--img-dir", type=str, default="", help="Đường dẫn thư mục ảnh KINS testing")
    parser.add_argument("--ann-file", type=str, default="", help="Đường dẫn file annotation update_test_2020.json")
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn file checkpoint đơn lẻ")
    parser.add_argument("--checkpoints", nargs="+", default=[], help="Danh sách một hoặc nhiều file checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Thư mục chứa các file .pth để đánh giá hàng loạt")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch (mặc định: 32)")
    parser.add_argument("--resize", type=int, default=224, help="Kích thước ảnh đầu vào (mặc định: 224)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân đoạn nhị phân (mặc định: 0.5)")
    parser.add_argument("--num-classes", type=int, default=20, help="Số lớp đối tượng KINS (mặc định: 20)")
    parser.add_argument("--num-workers", type=int, default=-1, help="Số workers DataLoader (mặc định: auto)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-samples", type=int, default=0, help="Giới hạn số mẫu đánh giá (0 = toàn bộ)")
    parser.add_argument("--output", type=str, default="results/kins_eval_mapr_standard.json", help="Đường dẫn lưu kết quả JSON")
    parser.add_argument("--output-md", type=str, default="results/kins_eval_mapr_standard.md", help="Đường dẫn lưu bảng Markdown")

    args = parser.parse_args()

    # Thiết bị
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Worker
    if args.num_workers < 0:
        num_workers = 0 if os.name == "nt" else 2
    else:
        num_workers = args.num_workers

    # Tìm đường dẫn dataset KINS
    kins_img_candidates = [
        args.img_dir,
        "/content/kitti_data/testing/image_2",
        "/content/testing/image_2",
        "/content/drive/MyDrive/amodal_shape_prediction/data/KINS/testing/image_2",
        "data/KINS/testing/image_2",
        os.path.join(ROOT_DIR, "data", "KINS", "testing", "image_2"),
    ]
    img_dir = find_path(args.img_dir, kins_img_candidates)

    kins_ann_candidates = [
        args.ann_file,
        "/content/drive/MyDrive/amodal_shape_prediction/data/KINS/update_test_2020.json",
        "/content/data/KINS/update_test_2020.json",
        "data/KINS/update_test_2020.json",
        os.path.join(ROOT_DIR, "data", "KINS", "update_test_2020.json"),
    ]
    ann_file = find_path(args.ann_file, kins_ann_candidates)

    print("=" * 80)
    print("🚀 BỘ ĐÁNH GIÁ CHUẨN KINS (MAPR INVISIBLE mIOU STANDARD)")
    print(f"🔧 Thiết bị: {device} | Workers: {num_workers} | Batch size: {args.batch_size}")
    print(f"📁 Thư mục ảnh KINS: {img_dir}")
    print(f"📄 File chú thích: {ann_file}")
    print("=" * 80)

    if not os.path.exists(img_dir) or not os.path.exists(ann_file):
        print(f"❌ Không tìm thấy dữ liệu KINS tại {img_dir} hoặc {ann_file}")
        sys.exit(1)

    # Nạp dataset KINS
    transform = A.Compose([A.Resize(args.resize, args.resize)])
    dataset = KINSDataset(img_dir=img_dir, ann_file=ann_file, transform=transform)

    if args.max_samples > 0 and args.max_samples < len(dataset):
        print(f"⚠️ Chế độ Subset: Đánh giá trên {args.max_samples:,}/{len(dataset):,} mẫu kiểm thử!")
        dataset = Subset(dataset, range(args.max_samples))
    else:
        print(f"✅ Đánh giá đầy đủ trên toàn bộ {len(dataset):,} mẫu kiểm thử KINS")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Thu thập danh sách checkpoint
    ckpt_list = []
    if args.checkpoint:
        ckpt_list.append(args.checkpoint)
    if args.checkpoints:
        ckpt_list.extend(args.checkpoints)
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        found = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pth")))
        ckpt_list.extend(found)

    # Lọc các checkpoint trùng lặp hoặc không tồn tại
    unique_ckpts = []
    for c in ckpt_list:
        if c and c not in unique_ckpts:
            if os.path.exists(c):
                unique_ckpts.append(c)
            else:
                print(f"⚠️ Checkpoint không tồn tại (bỏ qua): {c}")

    if not unique_ckpts:
        print("❌ Không có checkpoint hợp lệ nào được chỉ định! Dùng --checkpoint hoặc --checkpoints")
        sys.exit(1)

    print(f"\n📦 Tìm thấy {len(unique_ckpts)} checkpoint để đánh giá:")
    for idx, c in enumerate(unique_ckpts, 1):
        print(f"   [{idx}] {c}")
    print("-" * 80)

    records = []
    for idx, ckpt_path in enumerate(unique_ckpts, 1):
        print(f"\n[{idx}/{len(unique_ckpts)}] Đang xử lý: {os.path.basename(ckpt_path)}")
        start_time = time.time()

        try:
            model, model_type, total_params, gate_alpha = load_model(
                ckpt_path, num_classes=args.num_classes, device=device
            )
            print(f"   ▶ Kiến trúc: {model_type} ({total_params:,} tham số)")
            if gate_alpha is not None:
                print(f"   ▶ Cổng alpha đã học: {gate_alpha:.4f}")

            eval_res = evaluate_single_checkpoint(
                model=model,
                loader=loader,
                device=device,
                threshold=args.threshold,
                desc=f"Evaluating [{idx}/{len(unique_ckpts)}]"
            )
            elapsed = time.time() - start_time

            rec = {
                "checkpoint": ckpt_path,
                "model_type": model_type,
                "total_params": total_params,
                "gate_alpha": gate_alpha,
                "elapsed_seconds": round(elapsed, 2),
                **eval_res
            }
            records.append(rec)

            print(f"   ✅ Kết quả:")
            print(f"      - Overall mIoU:           {eval_res['overall_miou']:.2f}%")
            print(f"      - Invisible mIoU (F1):    {eval_res['invisible_miou']:.2f}% (Chuẩn MAPR)")
            print(f"      - Occlusion Recall (F2):  {eval_res['occlusion_recall']:.2f}% (Độ phủ cũ)")
            print(f"      - Precision (mAP):        {eval_res['precision']:.2f}%")
            print(f"      - Recall (mAR):           {eval_res['recall']:.2f}%")
            print(f"      - Thời gian:              {elapsed:.1f}s")

        except Exception as e:
            print(f"   ❌ Lỗi khi đánh giá {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()

    # Tạo báo cáo Markdown
    md_content = build_markdown_table(records)
    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ KINS (CHUẨN MAPR):")
    print("=" * 80)
    print(md_content)
    print("=" * 80)

    # Lưu kết quả ra file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": "KINS",
                "ann_file": ann_file,
                "img_dir": img_dir,
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"💾 Đã lưu kết quả JSON vào: {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐÁNH GIÁ CHUẨN KINS (MAPR INVISIBLE mIOU)\n\n")
        f.write(f"- Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- File chú thích: `{ann_file}`\n")
        f.write(f"- Thư mục ảnh: `{img_dir}`\n\n")
        f.write(md_content + "\n")
    print(f"💾 Đã lưu báo cáo Markdown vào: {args.output_md}")


if __name__ == "__main__":
    main()
