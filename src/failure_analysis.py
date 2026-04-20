"""
=============================================================================
FAILURE_ANALYSIS.PY - Phân tích Worst Cases (IoU < 30%)
=============================================================================
Script này phân tích những ảnh có IoU thấp (dưới 30%) để hiểu lý do model
thất bại. Hỗ trợ in số lượng worst cases và hiển thị 5 mẫu tồi nhất.

Sử dụng:
    python failure_analysis.py --eval-results results.json --output worst_cases.png
"""

import json, argparse, os, numpy as np, cv2, matplotlib.pyplot as plt, torch, albumentations as A
from tqdm import tqdm
from model import AmodalSwinUNet
from dataset import AmodalDataset


def calculate_occlusion_stats(target_mask, visible_mask):
    """Tính toán thống kê về vùng bị che khuất"""
    target_area = (target_mask > 0).sum()
    if target_area == 0:
        return 0.0
    occluded_area = (target_mask - visible_mask).clip(min=0).sum()
    return (occluded_area / target_area) * 100


def calculate_complexity(image, kernel_size=5):
    """Tính độ phức tạp hình ảnh"""
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image_gray = (image * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    variance = laplacian.var()
    return min(variance / 1000, 1.0)


def failure_analysis(args):
    """Phân tích và hiển thị worst cases"""
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Thiết bị: {device}")

    with open(args.eval_results, "r", encoding="utf-8") as f:
        results = json.load(f)
    per_sample_metrics = results["per_sample_metrics"]

    worst_cases = [
        (i, m["iou"])
        for i, m in enumerate(per_sample_metrics)
        if m["iou"] < args.failure_threshold
    ]
    worst_cases.sort(key=lambda x: x[1])

    num_worst = len(worst_cases)
    percentage = (num_worst / len(per_sample_metrics)) * 100

    print(f"Worst cases: {num_worst}/{len(per_sample_metrics)} ({percentage:.2f}%)")

    if num_worst == 0:
        print("Không tìm thấy worst cases!")
        return

    transform = A.Compose([A.Resize(args.resize, args.resize)])
    dataset = AmodalDataset(
        img_dir=args.img_dir, ann_file=args.ann_file, transform=transform
    )
    model = AmodalSwinUNet(num_classes=91).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    num_to_show = min(args.num_worst_show, num_worst)
    fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5 * num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    failure_details = []

    with torch.no_grad():
        for plot_idx, (sample_idx, iou) in enumerate(
            tqdm(worst_cases[:num_to_show], desc="Rendering")
        ):
            input_tensor, target_mask, occluded_region, class_id = dataset[sample_idx]
            input_batch = input_tensor.unsqueeze(0).to(device)
            class_id_batch = torch.tensor([class_id.item()]).to(device)

            output_logits = model(input_batch, class_id_batch)
            pred_mask = (
                (torch.sigmoid(output_logits) > args.threshold).squeeze().cpu().numpy()
            )

            img_rgb = np.clip(input_tensor[:3].numpy().transpose(1, 2, 0), 0, 1)
            visible_mask = input_tensor[3].numpy()
            truth_mask = target_mask.numpy()

            iou_score = per_sample_metrics[sample_idx]["iou"]
            dice_score = per_sample_metrics[sample_idx]["dice"]
            occlusion_pct = calculate_occlusion_stats(truth_mask, visible_mask)
            complexity = calculate_complexity(img_rgb)

            failure_details.append(
                {
                    "sample_idx": sample_idx,
                    "iou": iou_score,
                    "dice": dice_score,
                    "occlusion_percentage": occlusion_pct,
                    "complexity_score": complexity,
                    "has_occlusion": per_sample_metrics[sample_idx]["has_occlusion"],
                }
            )

            axes[plot_idx, 0].imshow(img_rgb)
            axes[plot_idx, 0].set_title(
                f"Original (#{sample_idx})", fontsize=10, fontweight="bold"
            )
            axes[plot_idx, 0].axis("off")

            axes[plot_idx, 1].imshow(truth_mask, cmap="gray", vmin=0, vmax=1)
            axes[plot_idx, 1].set_title(
                f"GT (Occ: {occlusion_pct:.1f}%)", fontsize=10, fontweight="bold"
            )
            axes[plot_idx, 1].axis("off")

            axes[plot_idx, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
            axes[plot_idx, 2].set_title(
                f"Pred (IoU: {iou_score*100:.2f}%)",
                fontsize=10,
                fontweight="bold",
                color="red",
            )
            axes[plot_idx, 2].axis("off")

    plt.tight_layout()
    output_path = args.output or "failure_analysis_worst_cases.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Lưu: {output_path}")

    if args.save_details:
        details_path = args.details_output or "failure_analysis_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump({"failure_cases": failure_details}, f, indent=2)

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Phân tích Worst Cases")
    parser.add_argument(
        "--eval-results", type=str, required=True, help="File JSON kết quả"
    )
    parser.add_argument("--img-dir", type=str, default="../data/val2014")
    parser.add_argument(
        "--ann-file", type=str, default="../data/annotations/COCO_amodal_val2014.json"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="../checkpoints/swin_amodal_epoch_30.pth"
    )
    parser.add_argument("--failure-threshold", type=float, default=0.3)
    parser.add_argument("--num-worst-show", type=int, default=5)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--details-output", type=str, default="")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    failure_analysis(args)
