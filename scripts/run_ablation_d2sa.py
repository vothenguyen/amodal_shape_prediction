"""
===================================================================================
RUN ABLATION D2SA - Chạy toàn bộ hoặc từng cấu hình ablation trên D2SA
===================================================================================
Script tự động hóa huấn luyện và đánh giá các cấu hình ablation trên D2SA:
- Row 1: 4ch (RGB+Vis) | No Edge | No Emb | No Spatial | BCE Loss
- Row 2: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | BCE Loss
- Row 3: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | Occ-Aware Loss (5x)
- Row 4: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | No Spatial | Occ-Aware Loss (5x) [Main]
- Row 5: 5ch (RGB+Vis+Edge) | Edge | No Emb | Spatial Attn | Occ-Aware Loss (5x)
- Row 6: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | Spatial Attn | Occ-Aware Loss (5x) [Full]
- Row 7: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | SGGatedBottleneck | Occ-Aware Loss (5x) [Proposed SG-GBA]

Ví dụ sử dụng:
  # Chạy thử smoke test 6 config trên subset 30 mẫu, 1 epoch:
  python scripts/run_ablation_d2sa.py --mode train --epochs 1 --subset-size 30 --batch-size 2

  # Chạy full huấn luyện Row 4 trên GPU:
  python scripts/run_ablation_d2sa.py --rows 4 --mode train --epochs 30 --device cuda

  # Chạy trên Google Colab lưu trực tiếp vào Drive:
  python scripts/run_ablation_d2sa.py --rows 4 --epochs 30 --batch-size 16 --accumulation-steps 1 --num-workers 8 --device cuda --drive-dir /content/drive/MyDrive/amodal_shape_prediction
===================================================================================
"""

import os
import sys
import json
import argparse
import subprocess

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

CONFIG_MAP = {
    1: {
        "name": "Row 1 (4ch, BCE, No Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce", "evaluate.py"),
        "ckpt_subdir": "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce",
        "log_subdir": "row1_bce_vis_noedge_noemb_nospatial",
        "eval_name": "row1_eval.json",
    },
    2: {
        "name": "Row 2 (5ch, BCE, Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "evaluate.py"),
        "ckpt_subdir": "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce",
        "log_subdir": "row2_bce_vis_edge_noemb_nospatial",
        "eval_name": "row2_eval.json",
    },
    3: {
        "name": "Row 3 (5ch, OccLoss 5x, Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding", "evaluate.py"),
        "ckpt_subdir": "amodal-shape-prediction-no-spatial-no-embeding",
        "log_subdir": "row3_occ_edge_noemb_nospatial",
        "eval_name": "row3_eval.json",
    },
    4: {
        "name": "Row 4 (Main Config: 5ch, OccLoss 5x, Edge, Emb 60, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "train_d2sa.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "evaluate_d2sa.py"),
        "ckpt_subdir": "amodal_shape_prediction_main_config",
        "log_subdir": "row4_main_config",
        "eval_name": "row4_eval.json",
    },
    5: {
        "name": "Row 5 (5ch, OccLoss 5x, Edge, No Emb, Spatial Attn)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_no_embeding", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_no_embeding", "evaluate.py"),
        "ckpt_subdir": "amodal_shape_prediction_no_embeding",
        "log_subdir": "row5_occ_edge_noemb_spatial",
        "eval_name": "row5_eval.json",
    },
    6: {
        "name": "Row 6 (Full Config: 5ch, OccLoss 5x, Edge, Emb 60, Spatial Attn)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_full_config", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_full_config", "evaluate.py"),
        "ckpt_subdir": "amodal_shape_prediction_full_config",
        "log_subdir": "row6_full_config",
        "eval_name": "row6_eval.json",
    },
    7: {
        "name": "Row 7 (Proposed SG-GBA: 5ch, OccLoss 5x, Edge, Emb 60, SGGatedBottleneck)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_sg_gba_bottleneck", "train_d2sa.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_sg_gba_bottleneck", "evaluate_d2sa.py"),
        "ckpt_subdir": "row7_sg_gba_bottleneck",
        "log_subdir": "row7_sg_gba_bottleneck",
        "eval_name": "row7_eval.json",
    },
}


def run_command(cmd, desc):
    print(f"\n{'='*75}")
    print(f"▶ {desc}")
    print(f"  Lệnh: {' '.join(cmd)}")
    print(f"{'='*75}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"❌ LỖI khi chạy {desc} (Exit code: {ret.returncode})")
        return False
    return True


def generate_table_v_summary(results_dir, executed_rows):
    """
    Tổng hợp kết quả đánh giá từ các file rowX_eval.json thành table_v_summary.json và table_v_summary.md
    """
    table_records = []
    for r in sorted(executed_rows):
        if r not in CONFIG_MAP:
            continue
        cfg = CONFIG_MAP[r]
        eval_path = os.path.join(results_dir, cfg["eval_name"])
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                sm = eval_data.get("summary_metrics", {})
                table_records.append({
                    "row": r,
                    "name": cfg["name"],
                    "mIoU": round(sm.get("mIoU", 0.0), 2),
                    "inv_mIoU": round(sm.get("invisible_mIoU", 0.0), 2),
                    "dice": round(sm.get("dice", 0.0), 2),
                    "precision": round(sm.get("precision", 0.0), 2),
                    "recall": round(sm.get("recall", 0.0), 2),
                })
            except Exception as e:
                print(f"⚠️ Không thể đọc file metrics Row {r} ({eval_path}): {e}")

    if not table_records:
        return

    # Lưu JSON
    json_path = os.path.join(results_dir, "table_v_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table_records, f, indent=2, ensure_ascii=False)
    print(f"📊 Đã tổng hợp dữ liệu Table V tại: {json_path}")

    # Lưu Markdown
    md_path = os.path.join(results_dir, "table_v_summary.md")
    lines = [
        "# 📊 BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ ABLATION STUDY D2SA (TABLE V)",
        "",
        "| Row | Cấu hình | mIoU (%) | Invisible mIoU (%) | Dice (%) | Precision (%) | Recall (%) |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rec in table_records:
        lines.append(
            f"| {rec['row']} | {rec['name']} | **{rec['mIoU']:.2f}** | **{rec['inv_mIoU']:.2f}** | {rec['dice']:.2f} | {rec['precision']:.2f} | {rec['recall']:.2f} |"
        )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📜 Đã xuất báo cáo Markdown tại: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Runner cho D2SA (7 cấu hình)")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7], help="Danh sách Row cần chạy (1–7)")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "all"], default="all", help="Chế độ: train, eval, hoặc all")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu huấn luyện (0: toàn bộ)")
    parser.add_argument("--val-subset-size", type=int, default=0, help="Số mẫu validation/đánh giá (0: toàn bộ)")
    parser.add_argument("--num-workers", type=int, default=2, help="Số workers cho DataLoader")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Epoch tiếp tục huấn luyện (0: từ đầu)")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Đường dẫn trực tiếp tới checkpoint cần resume")
    parser.add_argument("--drive-dir", type=str, default="", help="Đường dẫn thư mục Google Drive để lưu checkpoints, logs và results trực tiếp")
    args = parser.parse_args()

    python_exe = sys.executable

    # Thiết lập thư mục lưu trữ (Ưu tiên drive-dir nếu được truyền vào)
    base_dir = os.path.abspath(args.drive_dir) if args.drive_dir else ROOT_DIR
    base_ckpt_dir = os.path.join(base_dir, "checkpoints", "d2sa")
    base_log_dir = os.path.join(base_dir, "logs", "d2sa")
    base_results_dir = os.path.join(base_dir, "results", "d2sa")

    os.makedirs(base_ckpt_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(base_results_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("🏆 RUNNER ABLATION STUDY D2SA (MVTec D2S Amodal)")
    print(f"   Rows cần chạy: {args.rows}")
    print(f"   Chế độ:        {args.mode}")
    print(f"   Số epoch:      {args.epochs}")
    print(f"   Batch size:    {args.batch_size} (Tích lũy: {args.accumulation_steps})")
    print(f"   Train subset:  {args.subset_size if args.subset_size > 0 else 'Full dataset'}")
    print(f"   Val subset:    {args.val_subset_size if args.val_subset_size > 0 else 'Full dataset'}")
    print(f"   Num workers:   {args.num_workers}")
    if args.drive_dir:
        print(f"   Google Drive:  {args.drive_dir} (Checkpoints, logs và kết quả được lưu trực tiếp)")
    print("=" * 75)

    success_rows = []
    failed_rows = []

    for r in sorted(args.rows):
        if r not in CONFIG_MAP:
            print(f"⚠️ Row {r} không hợp lệ (chỉ hỗ trợ 1..7)")
            continue

        cfg = CONFIG_MAP[r]
        ckpt_dir = os.path.join(base_ckpt_dir, cfg["ckpt_subdir"])
        log_dir = os.path.join(base_log_dir, cfg["log_subdir"])
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        row_ok = True

        # TRAIN
        if args.mode in ["train", "all"]:
            train_cmd = [
                python_exe, cfg["train_script"],
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--accumulation-steps", str(args.accumulation_steps),
                "--num-workers", str(args.num_workers),
                "--lr", str(args.lr),
                "--device", args.device,
                "--checkpoint-dir", ckpt_dir,
                "--log-dir", log_dir,
            ]
            if args.subset_size > 0:
                train_cmd += ["--subset-size", str(args.subset_size)]
            if args.val_subset_size > 0:
                train_cmd += ["--val-subset-size", str(args.val_subset_size)]
            if args.resume_epoch > 0:
                train_cmd += ["--resume-epoch", str(args.resume_epoch)]
            if args.resume_checkpoint:
                train_cmd += ["--resume-checkpoint", args.resume_checkpoint]

            ok = run_command(train_cmd, f"Huấn luyện {cfg['name']}")
            if not ok:
                row_ok = False

        # EVAL
        if args.mode in ["eval", "all"] and row_ok:
            ckpt_path = os.path.join(ckpt_dir, f"swin_amodal_epoch_{args.epochs}.pth")
            eval_output = os.path.join(base_results_dir, cfg["eval_name"])
            eval_cmd = [
                python_exe, cfg["eval_script"],
                "--checkpoint", ckpt_path,
                "--output", eval_output,
                "--num-workers", str(args.num_workers),
                "--device", args.device,
            ]
            eval_subset = args.val_subset_size if args.val_subset_size > 0 else args.subset_size
            if eval_subset > 0:
                eval_cmd += ["--subset-size", str(eval_subset)]

            ok = run_command(eval_cmd, f"Đánh giá {cfg['name']}")
            if not ok:
                row_ok = False

        if row_ok:
            success_rows.append(r)
        else:
            failed_rows.append(r)

    # Tự động xuất tổng hợp Table V nếu có ít nhất 1 cấu hình đánh giá thành công
    if args.mode in ["eval", "all"] and success_rows:
        generate_table_v_summary(base_results_dir, success_rows)

    print("\n" + "=" * 75)
    print("📋 TỔNG KẾT TIẾN TRÌNH ABLATION D2SA")
    print("=" * 75)
    print(f"Thành công ({len(success_rows)}/{len(args.rows)}): {success_rows}")
    if failed_rows:
        print(f"Thất bại ({len(failed_rows)}/{len(args.rows)}):   {failed_rows}")
    print("\n📁 Thư mục lưu trữ Checkpoints:")
    for r in sorted(args.rows):
        if r in CONFIG_MAP:
            print(f"   - Row {r}: {os.path.join(base_ckpt_dir, CONFIG_MAP[r]['ckpt_subdir'])}")
    print("\n📁 Thư mục lưu trữ Logs & Metrics:")
    for r in sorted(args.rows):
        if r in CONFIG_MAP:
            print(f"   - Row {r}: {os.path.join(base_log_dir, CONFIG_MAP[r]['log_subdir'])}")
    print(f"\n📊 Kết quả đánh giá Table V: {base_results_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
