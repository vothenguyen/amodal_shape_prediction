"""
Script để phân tích phân phối occlusion trong dataset.
Giúp xác định:
1. Bao nhiêu phần trăm mẫu có occlusion
2. Phân phối occlusion ratio
3. Kích thước vùng bị che khuất
"""

import os
import cv2
import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_occlusion_dataset(img_dir, ann_file, output_dir="occlusion_analysis"):
    """
    Phân tích dataset occlusion một cách chi tiết.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"📊 Đang phân tích dataset: {ann_file}")
    coco = COCO(ann_file)

    # Thống kê
    stats = {
        "total_samples": 0,
        "occluded_samples": 0,  # Có occlusion > 0
        "heavily_occluded": defaultdict(int),  # ratio > threshold
        "occlusion_ratios": [],  # [0, 1]
        "occlusion_areas": [],  # pixel count
        "amodal_areas": [],  # amodal mask area
    }

    print("🔍 Bắt đầu duyệt dataset...")
    for ann_id, ann in tqdm(coco.anns.items(), desc="Analyzing"):
        if "regions" not in ann:
            continue

        img_id = ann["image_id"]
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        for region_idx, region in enumerate(ann["regions"]):
            if "segmentation" not in region:
                continue

            stats["total_samples"] += 1

            # ===== Vẽ AMODAL MASK =====
            amodal_mask = np.zeros((height, width), dtype=np.uint8)
            segs = region["segmentation"]
            if isinstance(segs, list) and len(segs) > 0:
                if isinstance(segs[0], (int, float)):
                    segs = [segs]
                for poly in segs:
                    if len(poly) >= 6:
                        poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(amodal_mask, [poly_2d], 1)

            # ===== Vẽ VISIBLE MASK =====
            visible_mask = amodal_mask.copy()
            target_order = region.get("order", 0)

            for other_region in ann["regions"]:
                other_order = other_region.get("order", 0)
                if other_order < target_order and "segmentation" in other_region:
                    other_segs = other_region["segmentation"]
                    if isinstance(other_segs, list) and len(other_segs) > 0:
                        if isinstance(other_segs[0], (int, float)):
                            other_segs = [other_segs]
                        for poly in other_segs:
                            if len(poly) >= 6:
                                poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(visible_mask, [poly_2d], 0)

            # ===== TÍNH CHỈ SỐ OCCLUSION =====
            amodal_area = amodal_mask.sum()
            visible_area = visible_mask.sum()
            occluded_area = max(0, amodal_area - visible_area)

            if amodal_area > 0:
                occlusion_ratio = occluded_area / amodal_area
            else:
                occlusion_ratio = 0

            stats["occlusion_ratios"].append(occlusion_ratio)
            stats["occlusion_areas"].append(occluded_area)
            stats["amodal_areas"].append(amodal_area)

            if occlusion_ratio > 0.01:  # Có occlusion ít nhất 1%
                stats["occluded_samples"] += 1

            # Đếm số mẫu ở các ngưỡng khác nhau
            for threshold in [0.1, 0.25, 0.5, 0.75]:
                if occlusion_ratio >= threshold:
                    stats["heavily_occluded"][f">{threshold}"] += 1

    # ===== IN KẾT QUẢ =====
    print("\n" + "=" * 70)
    print("📊 OCCLUSION DATASET ANALYSIS")
    print("=" * 70)
    print(f"📌 Tổng mẫu: {stats['total_samples']}")
    print(
        f"✅ Mẫu có occlusion (>1%): {stats['occluded_samples']} "
        f"({100*stats['occluded_samples']/stats['total_samples']:.1f}%)"
    )

    print("\n🎯 PHÂN PHỐI OCCLUSION RATIO:")
    for threshold in [0.1, 0.25, 0.5, 0.75]:
        count = stats["heavily_occluded"][f">{threshold}"]
        pct = 100 * count / stats["total_samples"] if stats["total_samples"] > 0 else 0
        print(f"  • Occlusion > {100*threshold:.0f}%: {count} mẫu ({pct:.1f}%)")

    # Thống kê occlusion ratio
    occlusion_ratios = np.array(stats["occlusion_ratios"])
    print("\n📈 THỐNG KÊ OCCLUSION RATIO:")
    print(f"  • Mean:   {occlusion_ratios.mean():.4f}")
    print(f"  • Median: {np.median(occlusion_ratios):.4f}")
    print(f"  • Max:    {occlusion_ratios.max():.4f}")
    print(f"  • Min:    {occlusion_ratios.min():.4f}")
    print(f"  • Std:    {occlusion_ratios.std():.4f}")

    # Thống kê kích thước
    amodal_areas = np.array(stats["amodal_areas"])
    occluded_areas = np.array(stats["occlusion_areas"])
    print("\n📏 THỐNG KÊ KÍCH THƯỚC AMODAL:")
    print(f"  • Mean:   {amodal_areas.mean():.0f} pixels")
    print(f"  • Median: {np.median(amodal_areas):.0f} pixels")

    print("\n📏 THỐNG KÊ KÍCH THƯỚC OCCLUDED:")
    non_zero_occ = occluded_areas[occluded_areas > 0]
    if len(non_zero_occ) > 0:
        print(f"  • Mean (non-zero):   {non_zero_occ.mean():.0f} pixels")
        print(f"  • Median (non-zero): {np.median(non_zero_occ):.0f} pixels")

    print("=" * 70)

    # ===== VẼ BIỂU ĐỒ =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram occlusion ratio
    axes[0, 0].hist(occlusion_ratios, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Occlusion Ratio")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Occlusion Ratio")
    axes[0, 0].axvline(occlusion_ratios.mean(), color="r", linestyle="--", label="Mean")
    axes[0, 0].legend()

    # 2. Cumulative distribution
    sorted_ratios = np.sort(occlusion_ratios)
    cumsum = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    axes[0, 1].plot(sorted_ratios, cumsum, linewidth=2)
    axes[0, 1].set_xlabel("Occlusion Ratio")
    axes[0, 1].set_ylabel("Cumulative %")
    axes[0, 1].set_title("Cumulative Distribution of Occlusion")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Bar chart: samples by threshold
    thresholds = [0, 0.1, 0.25, 0.5, 0.75]
    counts = []
    for i in range(len(thresholds) - 1):
        count = np.sum(
            (occlusion_ratios >= thresholds[i]) & (occlusion_ratios < thresholds[i + 1])
        )
        counts.append(count)
    count_final = np.sum(occlusion_ratios >= thresholds[-1])
    counts.append(count_final)

    labels = [
        f"{100*thresholds[i]:.0f}-{100*thresholds[i+1]:.0f}%"
        for i in range(len(thresholds) - 1)
    ]
    labels.append(f">{100*thresholds[-1]:.0f}%")
    axes[1, 0].bar(labels, counts, edgecolor="black", alpha=0.7)
    axes[1, 0].set_ylabel("Number of Samples")
    axes[1, 0].set_title("Samples by Occlusion Ratio Bins")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Occlusion area vs Amodal area
    axes[1, 1].scatter(amodal_areas, occluded_areas, alpha=0.3, s=10)
    axes[1, 1].set_xlabel("Amodal Area (pixels)")
    axes[1, 1].set_ylabel("Occluded Area (pixels)")
    axes[1, 1].set_title("Occlusion Area vs Amodal Area")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "occlusion_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n💾 Biểu đồ đã lưu tại: {output_path}")

    # ===== LƯU THỐNG KÊ =====
    stats_summary = {
        "total_samples": int(stats["total_samples"]),
        "occluded_samples": int(stats["occluded_samples"]),
        "occluded_percentage": float(
            stats["occluded_samples"] / stats["total_samples"] * 100
            if stats["total_samples"] > 0
            else 0
        ),
        "occlusion_ratio_stats": {
            "mean": float(occlusion_ratios.mean()),
            "median": float(np.median(occlusion_ratios)),
            "max": float(occlusion_ratios.max()),
            "min": float(occlusion_ratios.min()),
            "std": float(occlusion_ratios.std()),
        },
        "samples_above_threshold": {
            "10%": int(stats["heavily_occluded"].get(">0.1", 0)),
            "25%": int(stats["heavily_occluded"].get(">0.25", 0)),
            "50%": int(stats["heavily_occluded"].get(">0.5", 0)),
            "75%": int(stats["heavily_occluded"].get(">0.75", 0)),
        },
    }

    stats_path = os.path.join(output_dir, "occlusion_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"💾 Thống kê đã lưu tại: {stats_path}")

    return stats_summary


if __name__ == "__main__":
    import os

    # Lấy thư mục gốc của project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Phân tích training set
    print("\n🔴 PHÂN TÍCH TRAINING SET\n")
    analyze_occlusion_dataset(
        img_dir=os.path.join(project_root, "data/train2014"),
        ann_file=os.path.join(
            project_root, "data/annotations/COCO_amodal_train2014.json"
        ),
        output_dir=os.path.join(project_root, "results/occlusion_analysis_train"),
    )

    # Phân tích validation set
    print("\n🔵 PHÂN TÍCH VALIDATION SET\n")
    analyze_occlusion_dataset(
        img_dir=os.path.join(project_root, "data/val2014"),
        ann_file=os.path.join(
            project_root, "data/annotations/COCO_amodal_val2014.json"
        ),
        output_dir=os.path.join(project_root, "results/occlusion_analysis_val"),
    )
