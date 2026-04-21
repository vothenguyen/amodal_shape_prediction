"""
Automated Experiment Runner - Chạy tất cả các phương pháp để so sánh.

Chạy: python src/run_experiments.py --exp-names baseline,tuned_10x,tuned_15x,balanced,combo
"""

import os
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path


EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (5x weight)",
        "args": [
            "--loss-type",
            "original",
            "--occlusion-weight",
            "5.0",
            "--no-balanced-sampling",
            "--epochs",
            "30",
        ],
    },
    "tuned_10x": {
        "name": "Tuned 10x Weight",
        "args": [
            "--loss-type",
            "original",
            "--occlusion-weight",
            "10.0",
            "--no-balanced-sampling",
            "--epochs",
            "30",
        ],
    },
    "tuned_15x": {
        "name": "Tuned 15x Weight (RECOMMENDED)",
        "args": [
            "--loss-type",
            "original",
            "--occlusion-weight",
            "15.0",
            "--no-balanced-sampling",
            "--epochs",
            "30",
        ],
    },
    "tuned_20x": {
        "name": "Tuned 20x Weight",
        "args": [
            "--loss-type",
            "original",
            "--occlusion-weight",
            "20.0",
            "--no-balanced-sampling",
            "--epochs",
            "30",
        ],
    },
    "focal_10x": {
        "name": "Focal Loss 10x",
        "args": [
            "--loss-type",
            "focal",
            "--occlusion-weight",
            "10.0",
            "--focal-gamma",
            "2.0",
            "--no-balanced-sampling",
            "--epochs",
            "30",
        ],
    },
    "balanced_10": {
        "name": "Balanced Sampling (10% threshold, 2x oversample)",
        "args": [
            "--loss-type",
            "original",
            "--occlusion-weight",
            "10.0",
            "--use-balanced-sampling",
            "--occlusion-threshold",
            "0.1",
            "--oversample-ratio",
            "2.0",
            "--epochs",
            "30",
        ],
    },
    "balanced_25": {
        "name": "Balanced Sampling (25% threshold, 2.5x oversample)",
        "args": [
            "--loss-type",
            "original",
            "--occlusion-weight",
            "10.0",
            "--use-balanced-sampling",
            "--occlusion-threshold",
            "0.25",
            "--oversample-ratio",
            "2.5",
            "--epochs",
            "30",
        ],
    },
    "combo": {
        "name": "Best Combo (Balanced + Focal + 15x)",
        "args": [
            "--loss-type",
            "combo",
            "--occlusion-weight",
            "15.0",
            "--focal-gamma",
            "2.0",
            "--use-balanced-sampling",
            "--occlusion-threshold",
            "0.1",
            "--oversample-ratio",
            "2.0",
            "--epochs",
            "30",
        ],
    },
}


def run_experiment(exp_name, exp_config, script_dir, project_root):
    """
    Chạy 1 experiment.

    Args:
        exp_name: Tên experiment
        exp_config: Dict với keys 'name' và 'args'
        script_dir: Thư mục chứa train_balanced.py
        project_root: Thư mục gốc project

    Returns:
        status: "success" hoặc "failed"
        log_file: Path tới log file
    """
    exp_dir = os.path.join(project_root, "results", "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "train.log")

    print(f"\n{'='*70}")
    print(f"🚀 RUNNING: {exp_config['name']}")
    print(f"   ID: {exp_name}")
    print(f"   Log: {log_file}")
    print(f"{'='*70}")

    # Prepare command
    cmd = (
        [
            "python",
            os.path.join(script_dir, "train_balanced.py"),
        ]
        + exp_config["args"]
        + [
            "--checkpoint-dir",
            os.path.join(project_root, "checkpoints"),
            "--results-dir",
            exp_dir,
        ]
    )

    try:
        # Run training
        with open(log_file, "w") as log:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=None,
            )

        if result.returncode == 0:
            print(f"✅ SUCCESS: {exp_name}")
            return "success", log_file
        else:
            print(f"❌ FAILED: {exp_name} (Exit code: {result.returncode})")
            return "failed", log_file

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {exp_name}")
        return "timeout", log_file
    except Exception as e:
        print(f"💥 ERROR: {exp_name}\n{str(e)}")
        return "error", log_file


def evaluate_experiment(exp_name, exp_dir, project_root, checkpoint_epoch=30):
    """
    Chạy evaluation trên validation set.

    Args:
        exp_name: Tên experiment
        exp_dir: Thư mục chứa checkpoint
        project_root: Thư mục gốc
        checkpoint_epoch: Epoch để evaluate

    Returns:
        eval_results: Dict kết quả evaluation
    """
    from src.evaluate import evaluate as eval_fn, parse_args

    # Tìm checkpoint
    checkpoint_path = os.path.join(
        project_root, "checkpoints", f"swin_amodal_epoch_{checkpoint_epoch}.pth"
    )

    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None

    print(f"\n📊 Evaluating {exp_name}...")

    # Prepare args
    class Args:
        img_dir = os.path.join(project_root, "data/val2014")
        ann_file = os.path.join(
            project_root, "data/annotations/COCO_amodal_val2014.json"
        )
        checkpoint = checkpoint_path
        output = os.path.join(exp_dir, f"eval_results_epoch{checkpoint_epoch}.json")
        device = "auto"
        resize = 224
        batch_size = 4
        num_workers = 0
        threshold = 0.5

    args = Args()
    try:
        results = eval_fn(args)
        return results
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return None


def compare_results(experiments_summary):
    """
    In ra bảng so sánh kết quả.

    Args:
        experiments_summary: Dict {exp_name: results}
    """
    print("\n" + "=" * 100)
    print("📊 EXPERIMENT COMPARISON")
    print("=" * 100)

    print(
        f"{'Experiment':<30} {'Status':<10} {'mIoU':<10} {'Invisible mIoU':<15} {'Dice':<10}"
    )
    print("-" * 100)

    best_overall = None
    best_invisible = None
    best_overall_val = 0
    best_invisible_val = 0

    for exp_name, summary in experiments_summary.items():
        status = summary.get("status", "unknown")
        results = summary.get("results", {})

        miou = results.get("overall_mIoU", 0)
        inv_miou = results.get("invisible_mIoU", 0)
        dice = results.get("dice", 0)

        miou_str = f"{miou*100:.2f}%" if miou > 0 else "N/A"
        inv_str = f"{inv_miou*100:.2f}%" if inv_miou > 0 else "N/A"
        dice_str = f"{dice*100:.2f}%" if dice > 0 else "N/A"

        # Track best
        if miou > best_overall_val:
            best_overall = exp_name
            best_overall_val = miou
        if inv_miou > best_invisible_val:
            best_invisible = exp_name
            best_invisible_val = inv_miou

        marker = ""
        if exp_name == best_overall:
            marker += " 🏆"
        if exp_name == best_invisible:
            marker += " ⭐"

        print(
            f"{exp_name:<30} {status:<10} {miou_str:<10} {inv_str:<15} {dice_str:<10}{marker}"
        )

    print("=" * 100)
    if best_overall:
        print(f"🏆 Best Overall mIoU: {best_overall} ({best_overall_val*100:.2f}%)")
    if best_invisible:
        print(
            f"⭐ Best Invisible mIoU: {best_invisible} ({best_invisible_val*100:.2f}%)"
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments")
    parser.add_argument(
        "--exp-names",
        type=str,
        default="baseline,tuned_15x,balanced_10,combo",
        help="Comma-separated experiment names to run",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run evaluation",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip evaluation"
    )
    parser.add_argument(
        "--checkpoint-epoch", type=int, default=30, help="Epoch to evaluate"
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Parse experiment names
    exp_names = [e.strip() for e in args.exp_names.split(",")]

    # Validate experiment names
    invalid = set(exp_names) - set(EXPERIMENTS.keys())
    if invalid:
        print(f"❌ Unknown experiments: {invalid}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return

    # ===== TRAINING PHASE =====
    experiments_summary = {}

    if not args.skip_training:
        print(f"\n🔥 STARTING TRAINING PHASE")
        print(f"Experiments to run: {exp_names}")

        for exp_name in exp_names:
            exp_config = EXPERIMENTS[exp_name]
            status, log_file = run_experiment(
                exp_name, exp_config, script_dir, project_root
            )

            experiments_summary[exp_name] = {
                "status": status,
                "log_file": log_file,
                "results": None,
            }

    # ===== EVALUATION PHASE =====
    if not args.skip_evaluation:
        print(f"\n📊 STARTING EVALUATION PHASE")

        for exp_name in exp_names:
            exp_dir = os.path.join(project_root, "results", "experiments", exp_name)

            # Only evaluate if training was successful or skipped
            if (
                exp_name not in experiments_summary
                or experiments_summary[exp_name]["status"] == "success"
            ):
                results = evaluate_experiment(
                    exp_name,
                    exp_dir,
                    project_root,
                    checkpoint_epoch=args.checkpoint_epoch,
                )

                if exp_name in experiments_summary:
                    experiments_summary[exp_name]["results"] = results
                else:
                    experiments_summary[exp_name] = {
                        "status": "evaluated",
                        "results": results,
                    }

    # ===== COMPARISON =====
    if len(experiments_summary) > 0:
        compare_results(experiments_summary)

    # Save summary
    summary_file = os.path.join(
        project_root,
        "results",
        f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        # Convert to serializable format
        serializable_summary = {}
        for exp_name, summary in experiments_summary.items():
            serializable_summary[exp_name] = {
                "status": summary.get("status"),
                "log_file": summary.get("log_file"),
                "results": summary.get("results"),  # Assuming results is already dict
            }
        json.dump(serializable_summary, f, indent=2)

    print(f"\n💾 Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
