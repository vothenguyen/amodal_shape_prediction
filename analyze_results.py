import json
import numpy as np
from collections import Counter

# Load eval_results
with open("results/eval_results.json", "r") as f:
    eval_data = json.load(f)

per_sample = eval_data["per_sample_metrics"]
ious = [s["iou"] for s in per_sample]
dices = [s["dice"] for s in per_sample]
invisible_ious = [s["invisible_iou"] for s in per_sample if s["invisible_iou"] >= 0]
has_occlusion = [s["has_occlusion"] for s in per_sample]

# Filter out extreme values and NaN
ious_valid = [x for x in ious if x >= 0 and x <= 1]
dices_valid = [x for x in dices if x >= 0 and x <= 1]

print("=" * 80)
print("📊 ĐỊNH LƯỢNG (QUANTITATIVE EVALUATION)")
print("=" * 80)
print(f"\n📈 Overall mIoU: {eval_data['overall_mIoU']:.4f} (84.09%)")
print(f"📈 Overall Dice: {eval_data['dice']:.4f} (89.84%)")
print(f"📈 Invisible mIoU: {eval_data['invisible_mIoU']:.4f} (55.11%)")
print(f"📊 Tổng số mẫu: {eval_data['samples']:,}")

print("\n🔍 PHÂN PHỐI IoU:")
print(f"  - Min IoU: {min(ious_valid):.4f}")
print(f"  - Max IoU: {max(ious_valid):.4f}")
print(f"  - Mean IoU: {np.mean(ious_valid):.4f}")
print(f"  - Median IoU: {np.median(ious_valid):.4f}")
print(f"  - Std Dev: {np.std(ious_valid):.4f}")
print(f"  - Q1 (25%): {np.percentile(ious_valid, 25):.4f}")
print(f"  - Q3 (75%): {np.percentile(ious_valid, 75):.4f}")

# Performance tiers
excellent = len([x for x in ious_valid if x >= 0.9])
good = len([x for x in ious_valid if 0.7 <= x < 0.9])
moderate = len([x for x in ious_valid if 0.5 <= x < 0.7])
poor = len([x for x in ious_valid if 0.3 <= x < 0.5])
failing = len([x for x in ious_valid if x < 0.3])

print("\n🎯 PHÂN LOẠI HIỆU NĂNG:")
print(
    f"  - Xuất sắc (IoU ≥ 0.90): {excellent:,} ({100*excellent/len(ious_valid):.1f}%)"
)
print(f"  - Tốt (0.70 - 0.90):    {good:,} ({100*good/len(ious_valid):.1f}%)")
print(
    f"  - Trung bình (0.50 - 0.70): {moderate:,} ({100*moderate/len(ious_valid):.1f}%)"
)
print(f"  - Yếu (0.30 - 0.50):    {poor:,} ({100*poor/len(ious_valid):.1f}%)")
print(f"  - Thất bại (< 0.30):    {failing:,} ({100*failing/len(ious_valid):.1f}%)")

print("\n🌙 PHÂN TÍCH OCCLUSION:")
num_occluded = sum(has_occlusion)
print(
    f"  - Mẫu có che khuất: {num_occluded:,} ({100*num_occluded/len(has_occlusion):.1f}%)"
)
print(
    f"  - Mẫu không che khuất: {len(has_occlusion)-num_occluded:,} ({100*(len(has_occlusion)-num_occluded)/len(has_occlusion):.1f}%)"
)

# Performance by occlusion
occluded_ious = [
    per_sample[i]["iou"]
    for i in range(len(per_sample))
    if per_sample[i]["has_occlusion"] and per_sample[i]["iou"] <= 1
]
non_occluded_ious = [
    per_sample[i]["iou"]
    for i in range(len(per_sample))
    if not per_sample[i]["has_occlusion"] and per_sample[i]["iou"] <= 1
]

print(f"\n  - Avg IoU (có che khuất): {np.mean(occluded_ious):.4f}")
print(f"  - Avg IoU (không che khuất): {np.mean(non_occluded_ious):.4f}")
print(f"  - Chênh lệch: {abs(np.mean(non_occluded_ious) - np.mean(occluded_ious)):.4f}")

print("\n" + "=" * 80)
print("📋 PHÂN TÍCH FAILURE CASES")
print("=" * 80)

# Load failure details
with open("results/failure_details.json", "r") as f:
    failure_data = json.load(f)

failure_cases = failure_data["failure_cases"]
occlusion_percentages = [f["occlusion_percentage"] for f in failure_cases]
complexity_scores = [f["complexity_score"] for f in failure_cases]

print(f"\n🔴 Top 5 worst cases:")
for i, case in enumerate(failure_cases):
    print(
        f"  {i+1}. Sample {case['sample_idx']}: Occlusion {case['occlusion_percentage']:.0f}%, Complexity {case['complexity_score']:.2f}"
    )

print("\n" + "=" * 80)
print("⚡ PHÂN TÍCH ABLATION STUDY")
print("=" * 80)

# Load ablation results
with open("results/ablation_results.json", "r") as f:
    ablation_data = json.load(f)

print(f"\n🔍 Spatial Attention Module Impact:")
print(f"  - WITH Attention: {ablation_data['with_attention']:.6f}")
print(f"  - WITHOUT Attention: {ablation_data['without_attention']:.6f}")
print(
    f"  - Improvement: {ablation_data['improvement_percent']:.4f}% ({ablation_data['improvement_percent']/100:.6f})"
)

print("\n" + "=" * 80)
