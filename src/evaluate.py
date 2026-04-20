import argparse
import json
import os

import albumentations as A
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AmodalSwinUNet
from dataset import AmodalDataset


def calculate_metrics(pred_logits, target, visible, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    dice = (2.0 * intersection + 1e-6) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
    )

    invisible_target = torch.clamp(target - visible, min=0.0)
    pred_invisible = pred * (invisible_target > 0).float()

    inv_intersection = (pred_invisible * invisible_target).sum(dim=(2, 3))
    inv_union = (
        pred_invisible.sum(dim=(2, 3))
        + invisible_target.sum(dim=(2, 3))
        - inv_intersection
    )
    valid_mask = (invisible_target.sum(dim=(2, 3)) > 0).float()
    inv_iou = (inv_intersection + 1e-6) / (inv_union + 1e-6)

    return iou, dice, inv_iou, valid_mask


def build_transform(resize):
    return A.Compose([A.Resize(resize, resize)])


def evaluate(args):
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🔍 Đang đánh giá trên thiết bị: {device}")

    transform = build_transform(args.resize)
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = AmodalSwinUNet(num_classes=91).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    total_inv_iou = 0.0
    total_valid_inv = 0.0

    # Lưu per-sample metrics cho qualitative/failure analysis
    per_sample_metrics = []

    print("Đang chấm điểm từng pixel... Xin chờ trong giây lát!")
    with torch.no_grad():
        for inputs, targets, occluded_region, class_ids in tqdm(
            loader, desc="Evaluating"
        ):
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            visible_masks = inputs[:, 3:4, :, :].float()
            class_ids = class_ids.to(device)

            outputs = model(inputs, class_ids)
            iou, dice, inv_iou, valid_mask = calculate_metrics(
                outputs, targets, visible_masks, threshold=args.threshold
            )

            total_iou += iou.sum().item()
            total_dice += dice.sum().item()
            total_inv_iou += (inv_iou * valid_mask).sum().item()
            total_valid_inv += valid_mask.sum().item()

            # Lưu per-sample metrics
            for i in range(iou.shape[0]):
                per_sample_metrics.append(
                    {
                        "iou": iou[i].item(),
                        "dice": dice[i].item(),
                        "invisible_iou": (
                            inv_iou[i].item() if valid_mask[i].item() > 0 else -1.0
                        ),
                        "has_occlusion": valid_mask[i].item() > 0,
                    }
                )

    n_samples = len(dataset)
    m_iou = total_iou / n_samples if n_samples > 0 else 0.0
    m_dice = total_dice / n_samples if n_samples > 0 else 0.0
    m_inv_iou = total_inv_iou / total_valid_inv if total_valid_inv > 0 else 0.0

    print("\n" + "=" * 60)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"🔸 Dataset             : {args.ann_file}")
    print(f"🔸 Checkpoint          : {args.checkpoint}")
    print(f"🔸 Samples             : {n_samples}")
    print(f"🔸 Overall mIoU        : {m_iou * 100:.2f} %")
    print(f"🔸 Dice Coefficient    : {m_dice * 100:.2f} %")
    print(f"🔸 Invisible mIoU      : {m_inv_iou * 100:.2f} %")
    print("=" * 60)

    results = {
        "dataset": args.ann_file,
        "checkpoint": args.checkpoint,
        "samples": n_samples,
        "overall_mIoU": m_iou,
        "dice": m_dice,
        "invisible_mIoU": m_inv_iou,
        "threshold": args.threshold,
        "resize": args.resize,
        "device": str(device),
        "per_sample_metrics": per_sample_metrics,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Kết quả đã lưu tại: {args.output}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate amodal segmentation model")
    parser.add_argument(
        "--img-dir",
        type=str,
        default="../data/val2014",
        help="Thư mục chứa ảnh đánh giá",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="../data/annotations/COCO_amodal_val2014.json",
        help="File annotation COCOA/COCO amodal",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/swin_amodal_epoch_30.pth",
        help="Đường dẫn đến checkpoint của model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Kích thước batch đánh giá"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Số worker cho DataLoader"
    )
    parser.add_argument(
        "--resize", type=int, default=224, help="Kích thước resize chuẩn cho model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng để chuyển logits thành mask nhị phân",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Thiết bị chạy: auto, cpu hoặc cuda"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Lưu kết quả đánh giá ra file JSON"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
