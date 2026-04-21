"""
Visualization & Comparison Script
So sánh kết quả từ các experiment khác nhau.

Chạy: python src/compare_experiments.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_evaluation_results(results_dir):
    """Load tất cả kết quả evaluation từ experiments."""
    experiments = {}

    experiments_path = Path(results_dir) / "experiments"
    if not experiments_path.exists():
        print(f"❌ Experiments directory not found: {experiments_path}")
        return experiments

    for exp_folder in experiments_path.iterdir():
        if not exp_folder.is_dir():
            continue

        exp_name = exp_folder.name

        # Tìm eval_results*.json
        eval_files = list(exp_folder.glob("eval_results*.json"))
        if not eval_files:
            print(f"⚠️  No eval_results found in {exp_name}")
            continue

        eval_file = eval_files[0]  # Take first one

        try:
            with open(eval_file, "r") as f:
                results = json.load(f)
                experiments[exp_name] = results
                print(f"✅ Loaded: {exp_name}")
        except Exception as e:
            print(f"❌ Failed to load {eval_file}: {e}")

    return experiments


def plot_comparison(experiments, output_path="comparison.png"):
    """Vẽ biểu đồ so sánh các experiment."""

    if not experiments:
        print("❌ No experiments to compare")
        return

    # Extract metrics
    exp_names = list(experiments.keys())
    overall_miou = [
        experiments[name].get("overall_mIoU", 0) * 100 for name in exp_names
    ]
    invisible_miou = [
        experiments[name].get("invisible_mIoU", 0) * 100 for name in exp_names
    ]
    dice_scores = [experiments[name].get("dice", 0) * 100 for name in exp_names]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Overall mIoU
    colors_1 = ["green" if v == max(overall_miou) else "skyblue" for v in overall_miou]
    axes[0].bar(exp_names, overall_miou, color=colors_1, edgecolor="black", alpha=0.7)
    axes[0].set_title("Overall mIoU (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("mIoU (%)")
    axes[0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(overall_miou):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")

    # Plot 2: Invisible mIoU (MOST IMPORTANT)
    colors_2 = [
        "red" if v == max(invisible_miou) else "lightcoral" for v in invisible_miou
    ]
    axes[1].bar(exp_names, invisible_miou, color=colors_2, edgecolor="black", alpha=0.7)
    axes[1].set_title(
        "Invisible mIoU (%) ⭐", fontsize=12, fontweight="bold", color="red"
    )
    axes[1].set_ylabel("Invisible mIoU (%)")
    axes[1].tick_params(axis="x", rotation=45)
    for i, v in enumerate(invisible_miou):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")

    # Plot 3: Dice Coefficient
    colors_3 = ["orange" if v == max(dice_scores) else "moccasin" for v in dice_scores]
    axes[2].bar(exp_names, dice_scores, color=colors_3, edgecolor="black", alpha=0.7)
    axes[2].set_title("Dice Coefficient (%)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Dice (%)")
    axes[2].tick_params(axis="x", rotation=45)
    for i, v in enumerate(dice_scores):
        axes[2].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"💾 Comparison plot saved: {output_path}")
    plt.close()


def print_detailed_comparison(experiments):
    """In ra bảng so sánh chi tiết."""

    if not experiments:
        print("❌ No experiments to compare")
        return

    print("\n" + "=" * 120)
    print("📊 DETAILED EXPERIMENT COMPARISON")
    print("=" * 120)

    exp_names = list(experiments.keys())

    # Header
    print(
        f"{'Experiment':<25} {'Overall mIoU':<15} {'Invisible mIoU':<18} {'Dice':<15} {'Samples':<10}"
    )
    print("-" * 120)

    baseline_miou = None
    baseline_invisible = None

    for exp_name in sorted(exp_names):
        results = experiments[exp_name]

        overall = results.get("overall_mIoU", 0)
        invisible = results.get("invisible_mIoU", 0)
        dice = results.get("dice", 0)
        samples = results.get("samples", 0)

        # Calculate improvement over baseline
        if "baseline" in exp_name and baseline_miou is None:
            baseline_miou = overall
            baseline_invisible = invisible
            improve_overall = ""
            improve_invisible = ""
        else:
            if baseline_miou is not None:
                improve_overall = f"({(overall-baseline_miou)*100:+.1f}%)"
                improve_invisible = f"({(invisible-baseline_invisible)*100:+.1f}%)"
            else:
                improve_overall = ""
                improve_invisible = ""

        print(
            f"{exp_name:<25} {overall*100:>6.2f}% {improve_overall:<7} {invisible*100:>6.2f}% {improve_invisible:<10} {dice*100:>6.2f}% {samples:>10}"
        )

    print("=" * 120)


def create_summary_table(experiments, output_file="experiment_comparison.json"):
    """Lưu bảng so sánh dưới dạng JSON."""

    summary = {"timestamp": str(np.datetime64("today")), "experiments": {}}

    for exp_name, results in experiments.items():
        summary["experiments"][exp_name] = {
            "overall_mIoU": float(results.get("overall_mIoU", 0)),
            "invisible_mIoU": float(results.get("invisible_mIoU", 0)),
            "dice": float(results.get("dice", 0)),
            "samples": int(results.get("samples", 0)),
        }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary table saved: {output_file}")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    print(f"\n🔍 Loading experiments from: {results_dir}/experiments/")

    # Load experiments
    experiments = load_evaluation_results(results_dir)

    if not experiments:
        print("❌ No experiments found")
        return

    print(f"\n✅ Loaded {len(experiments)} experiments\n")

    # Print comparison
    print_detailed_comparison(experiments)

    # Plot comparison
    comparison_plot = os.path.join(results_dir, "experiment_comparison.png")
    plot_comparison(experiments, output_path=comparison_plot)

    # Create summary table
    comparison_table = os.path.join(results_dir, "experiment_comparison.json")
    create_summary_table(experiments, output_file=comparison_table)

    print("\n🎉 Comparison complete!")


if __name__ == "__main__":
    main()
