#!/usr/bin/env python
"""
MASTER EVALUATION SCRIPT
========================
Chạy toàn bộ quy trình đánh giá: Quantitative -> Qualitative -> Failure Analysis -> Ablation Study
"""

import os
import json
import subprocess
import sys


def run_command(cmd, description):
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Lỗi khi chạy: {description}")
        return False
    print(f"✅ Hoàn tất: {description}")
    return True


def main():
    # Config
    img_dir = "data/val2014"
    ann_file = "data/annotations/COCO_amodal_val2014.json"
    checkpoint = "checkpoints/swin_amodal_epoch_30.pth"
    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🎯 MASTER EVALUATION PIPELINE")
    print("=" * 70)

    # Step 1: Quantitative Evaluation
    eval_json = f"{results_dir}/eval_results.json"
    cmd1 = f"cd src && python evaluate.py --img-dir ../{img_dir} --ann-file ../{ann_file} --checkpoint ../{checkpoint} --batch-size 16 --num-workers 0 --output ../{eval_json}"
    if not run_command(cmd1, "1. Quantitative Evaluation"):
        return

    # Step 2: Qualitative Evaluation (Best Cases)
    qualitative_png = f"{results_dir}/qualitative_best_cases.png"
    cmd2 = f"cd src && python qualitative_eval.py --eval-results ../{eval_json} --img-dir ../{img_dir} --ann-file ../{ann_file} --checkpoint ../{checkpoint} --top-k 8 --output ../{qualitative_png}"
    if not run_command(cmd2, "2. Qualitative Evaluation (Top-8 Best Cases)"):
        return

    # Step 3: Failure Analysis (Worst Cases)
    failure_png = f"{results_dir}/failure_worst_cases.png"
    cmd3 = f"cd src && python failure_analysis.py --eval-results ../{eval_json} --img-dir ../{img_dir} --ann-file ../{ann_file} --checkpoint ../{checkpoint} --failure-threshold 0.3 --num-worst-show 5 --output ../{failure_png} --save-details --details-output ../{results_dir}/failure_details.json"
    if not run_command(cmd3, "3. Failure Analysis (Worst Cases)"):
        return

    # Step 4: Ablation Study
    ablation_json = f"{results_dir}/ablation_results.json"
    cmd4 = f"cd src && python ablation_study.py --img-dir ../{img_dir} --ann-file ../{ann_file} --checkpoint ../{checkpoint} --batch-size 16 --num-workers 0 --output ../{ablation_json}"
    if not run_command(cmd4, "4. Ablation Study (Spatial Attention)"):
        return

    # Summary
    print(f"\n{'='*70}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Evaluation JSON:        {eval_json}")
    print(f"✅ Qualitative PNG:        {qualitative_png}")
    print(f"✅ Failure Analysis PNG:   {failure_png}")
    print(f"✅ Failure Details JSON:   {results_dir}/failure_details.json")
    print(f"✅ Ablation JSON:          {ablation_json}")
    print(f"\n📁 Tất cả kết quả được lưu trong thư mục: {results_dir}/")
    print("\n🎨 Tiếp theo: Sử dụng những file này để trình bày báo cáo!")


if __name__ == "__main__":
    main()
