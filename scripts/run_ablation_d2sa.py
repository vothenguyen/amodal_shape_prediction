"""
===================================================================================
RUN ABLATION D2SA - Chạy toàn bộ hoặc từng cấu hình ablation trên D2SA
Tối ưu cho Google Colab / Máy chủ NVIDIA A100 (40GB/80GB VRAM) & Lưu Google Drive
===================================================================================
Script tự động hóa huấn luyện và đánh giá 6 cấu hình ablation trên D2SA:
- Row 1: 4ch (RGB+Vis) | No Edge | No Emb | No Spatial | BCE Loss
- Row 2: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | BCE Loss
- Row 3: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | Occ-Aware Loss (5x)
- Row 4: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | No Spatial | Occ-Aware Loss (5x) [Main]
- Row 5: 5ch (RGB+Vis+Edge) | Edge | No Emb | Spatial Attn | Occ-Aware Loss (5x)
- Row 6: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | Spatial Attn | Occ-Aware Loss (5x) [Full]

Ví dụ sử dụng trên Colab A100:
  # Chạy 1 lần toàn bộ 6 cấu hình (Batch 16, 30 epochs) lưu trực tiếp vào Google Drive:
  python scripts/run_ablation_d2sa.py --mode all --epochs 30 --batch-size 16 --accumulation-steps 1 --num-workers 8 --drive-dir /content/drive/MyDrive/amodal_shape_prediction

  # Chạy thử smoke test nhanh 1 epoch trên 50 mẫu:
  python scripts/run_ablation_d2sa.py --mode all --epochs 1 --subset-size 50 --batch-size 16

  # Đánh giá lại toàn bộ 6 cấu hình:
  python scripts/run_ablation_d2sa.py --mode eval --device cuda
===================================================================================
"""

import os
import sys
import json
import argparse
import subprocess

# Đảm bảo UTF-8 trên console
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
        "ckpt_subdir": os.path.join("d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce"),
        "log_subdir": os.path.join("d2sa", "row1_bce_vis_noedge_noemb_nospatial"),
        "eval_filename": "row1_eval.json",
        "input_ch": "4ch (RGB+Vis)",
        "edge": "❌",
        "emb": "❌",
        "spatial": "❌",
        "loss": "BCE",
    },
    2: {
        "name": "Row 2 (5ch, BCE, Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "evaluate.py"),
        "ckpt_subdir": os.path.join("d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce"),
        "log_subdir": os.path.join("d2sa", "row2_bce_vis_edge_noemb_nospatial"),
        "eval_filename": "row2_eval.json",
        "input_ch": "5ch (RGB+Vis+Edge)",
        "edge": "✅",
        "emb": "❌",
        "spatial": "❌",
        "loss": "BCE",
    },
    3: {
        "name": "Row 3 (5ch, OccLoss 5x, Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding", "evaluate.py"),
        "ckpt_subdir": os.path.join("d2sa", "amodal-shape-prediction-no-spatial-no-embeding"),
        "log_subdir": os.path.join("d2sa", "row3_occ_edge_noemb_nospatial"),
        "eval_filename": "row3_eval.json",
        "input_ch": "5ch (RGB+Vis+Edge)",
        "edge": "✅",
        "emb": "❌",
        "spatial": "❌",
        "loss": "Occ (5x) + Dice",
    },
    4: {
        "name": "Row 4 (Main Config: 5ch, OccLoss 5x, Edge, Emb 60, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "train_d2sa.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "evaluate_d2sa.py"),
        "ckpt_subdir": os.path.join("d2sa", "amodal_shape_prediction_main_config"),
        "log_subdir": os.path.join("d2sa", "row4_main_config"),
        "eval_filename": "row4_eval.json",
        "input_ch": "5ch (RGB+Vis+Edge)",
        "edge": "✅",
        "emb": "✅ (60)",
        "spatial": "❌",
        "loss": "Occ (5x) + Dice",
    },
    5: {
        "name": "Row 5 (5ch, OccLoss 5x, Edge, No Emb, Spatial Attn)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_no_embeding", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_no_embeding", "evaluate.py"),
        "ckpt_subdir": os.path.join("d2sa", "amodal_shape_prediction_no_embeding"),
        "log_subdir": os.path.join("d2sa", "row5_occ_edge_noemb_spatial"),
        "eval_filename": "row5_eval.json",
        "input_ch": "5ch (RGB+Vis+Edge)",
        "edge": "✅",
        "emb": "❌",
        "spatial": "✅ (k=7)",
        "loss": "Occ (5x) + Dice",
    },
    6: {
        "name": "Row 6 (Full Config: 5ch, OccLoss 5x, Edge, Emb 60, Spatial Attn)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_full_config", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_full_config", "evaluate.py"),
        "ckpt_subdir": os.path.join("d2sa", "amodal_shape_prediction_full_config"),
        "log_subdir": os.path.join("d2sa", "row6_full_config"),
        "eval_filename": "row6_eval.json",
        "input_ch": "5ch (RGB+Vis+Edge)",
        "edge": "✅",
        "emb": "✅ (60)",
        "spatial": "✅ (k=7)",
        "loss": "Occ (5x) + Dice",
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


def generate_table_v_summary(rows_to_check, base_results_dir):
    """Tổng hợp kết quả đánh giá thành bảng so sánh Table V chuẩn Markdown và JSON."""
    summary_data = []

    for r in sorted(rows_to_check):
        if r not in CONFIG_MAP:
            continue
        cfg = CONFIG_MAP[r]
        eval_path = os.path.join(base_results_dir, "d2sa", cfg["eval_filename"])
        if not os.path.exists(eval_path):
            eval_path = os.path.join(ROOT_DIR, "results", "d2sa", cfg["eval_filename"])

        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metrics = data.get("summary_metrics", {})
                summary_data.append({
                    "row": r,
                    "name": cfg["name"],
                    "input": cfg["input_ch"],
                    "edge": cfg["edge"],
                    "emb": cfg["emb"],
                    "spatial": cfg["spatial"],
                    "loss": cfg["loss"],
                    "mIoU": metrics.get("mIoU", 0.0),
                    "dice": metrics.get("dice", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "inv_mIoU": metrics.get("invisible_mIoU", 0.0),
                })
            except Exception as e:
                print(f"⚠️ Không thể đọc file kết quả {eval_path}: {e}")

    if not summary_data:
        return

    # In bảng Markdown
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM TABLE V (MVTec D2SA)")
    lines.append("=" * 90)
    lines.append("| Row | Cấu hình | Input | Edge | Cat Emb | Spatial | Loss | mIoU (%) | Dice (%) | Prec (%) | Rec (%) | Inv-mIoU (%) |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for item in summary_data:
        lines.append(
            f"| **{item['row']}** | {item['name']} | {item['input']} | {item['edge']} | {item['emb']} | {item['spatial']} | {item['loss']} | **{item['mIoU']:.2f}** | {item['dice']:.2f} | {item['precision']:.2f} | {item['recall']:.2f} | **{item['inv_mIoU']:.2f}** |"
        )
    lines.append("=" * 90)

    summary_text = "\n".join(lines)
    print(summary_text)

    # Lưu file
    os.makedirs(os.path.join(base_results_dir, "d2sa"), exist_ok=True)
    summary_md_path = os.path.join(base_results_dir, "d2sa", "table_v_summary.md")
    summary_json_path = os.path.join(base_results_dir, "d2sa", "table_v_summary.json")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Table V: Ablation Study on D2SA Dataset\n\n" + summary_text + "\n")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Đã lưu bảng tổng hợp Table V tại:\n   - {summary_md_path}\n   - {summary_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Runner cho D2SA trên NVIDIA A100 (40GB/80GB VRAM) / Colab")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="Danh sách Row cần chạy (1–6)")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "all"], default="all", help="Chế độ: train, eval, hoặc all")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện (mặc định 30)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (Mặc định 16 tối ưu cho A100 40GB/80GB VRAM)")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps (Mặc định 1 trên A100)")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu huấn luyện (0: toàn bộ)")
    parser.add_argument("--val-subset-size", type=int, default=0, help="Số mẫu validation định kỳ trong lúc train (0: toàn bộ)")
    parser.add_argument("--eval-subset-size", type=int, default=0, help="Số mẫu đánh giá cuối cùng (0: toàn bộ 15,654 mẫu)")
    parser.add_argument("--num-workers", type=int, default=8, help="Số workers cho DataLoader (Mặc định 8 cho Colab High-RAM)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (mặc định 1e-4)")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Epoch tiếp tục huấn luyện (0: từ đầu)")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Đường dẫn trực tiếp tới checkpoint cần resume")
    parser.add_argument("--drive-dir", type=str, default="", help="Đường dẫn thư mục Google Drive (VD: /content/drive/MyDrive/amodal_shape_prediction)")
    args = parser.parse_args()

    python_exe = sys.executable

    # Xác định thư mục lưu trữ cơ sở (local hoặc Google Drive)
    base_ckpt_dir = os.path.join(args.drive_dir, "checkpoints") if args.drive_dir else os.path.join(ROOT_DIR, "checkpoints")
    base_log_dir = os.path.join(args.drive_dir, "logs") if args.drive_dir else os.path.join(ROOT_DIR, "logs")
    base_results_dir = os.path.join(args.drive_dir, "results") if args.drive_dir else os.path.join(ROOT_DIR, "results")

    print("\n" + "=" * 75)
    print("🏆 RUNNER ABLATION STUDY D2SA (NVIDIA A100 40GB/80GB VRAM / COLAB)")
    print(f"   Rows cần chạy: {args.rows}")
    print(f"   Chế độ:        {args.mode}")
    print(f"   Số epoch:      {args.epochs}")
    print(f"   Batch size:    {args.batch_size} (Tích lũy: {args.accumulation_steps} -> Effective = {args.batch_size * args.accumulation_steps})")
    print(f"   Train subset:  {args.subset_size if args.subset_size > 0 else 'Full dataset (28,720 instances)'}")
    print(f"   Val định kỳ:   {args.val_subset_size if args.val_subset_size > 0 else 'Full val'}")
    print(f"   Eval cuối:     {args.eval_subset_size if args.eval_subset_size > 0 else 'Full validation (15,654 instances)'}")
    print(f"   Num workers:   {args.num_workers}")
    if args.drive_dir:
        print(f"   Google Drive:  {args.drive_dir}")
        print(f"   Checkpoints:   {base_ckpt_dir}/d2sa/")
        print(f"   Logs:          {base_log_dir}/d2sa/")
        print(f"   Results:       {base_results_dir}/d2sa/")
    print("=" * 75)

    success_rows = []
    failed_rows = []

    for r in sorted(args.rows):
        if r not in CONFIG_MAP:
            print(f"⚠️ Row {r} không hợp lệ (chỉ hỗ trợ 1..6)")
            continue

        cfg = CONFIG_MAP[r]
        row_ckpt_dir = os.path.join(base_ckpt_dir, cfg["ckpt_subdir"])
        row_log_dir = os.path.join(base_log_dir, cfg["log_subdir"])
        row_eval_out = os.path.join(base_results_dir, "d2sa", cfg["eval_filename"])

        os.makedirs(row_ckpt_dir, exist_ok=True)
        os.makedirs(row_log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(row_eval_out), exist_ok=True)

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
                "--checkpoint-dir", row_ckpt_dir,
                "--log-dir", row_log_dir,
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
            # Tìm checkpoint: ưu tiên epoch_30, fallback về last.pth
            ckpt_path = os.path.join(row_ckpt_dir, f"swin_amodal_epoch_{args.epochs}.pth")
            if not os.path.exists(ckpt_path):
                last_path = os.path.join(row_ckpt_dir, "last.pth")
                if os.path.exists(last_path):
                    ckpt_path = last_path

            eval_cmd = [
                python_exe, cfg["eval_script"],
                "--checkpoint", ckpt_path,
                "--num-workers", str(args.num_workers),
                "--device", args.device,
                "--output", row_eval_out,
            ]
            if args.eval_subset_size > 0:
                eval_cmd += ["--subset-size", str(args.eval_subset_size)]
            elif args.mode == "eval" and args.subset_size > 0:
                eval_cmd += ["--subset-size", str(args.subset_size)]

            ok = run_command(eval_cmd, f"Đánh giá {cfg['name']}")
            if not ok:
                row_ok = False

        if row_ok:
            success_rows.append(r)
        else:
            failed_rows.append(r)

    print("\n" + "=" * 75)
    print("📋 TỔNG KẾT TIẾN TRÌNH ABLATION D2SA")
    print("=" * 75)
    print(f"Thành công ({len(success_rows)}/{len(args.rows)}): {success_rows}")
    if failed_rows:
        print(f"Thất bại ({len(failed_rows)}/{len(args.rows)}):   {failed_rows}")

    # Tạo bảng tổng hợp Table V nếu có chạy đánh giá
    if args.mode in ["eval", "all"] and success_rows:
        generate_table_v_summary(success_rows, base_results_dir)

    print("\n📁 Thư mục lưu trữ Checkpoints:")
    for r in sorted(args.rows):
        if r in CONFIG_MAP:
            print(f"   - Row {r}: {os.path.join(base_ckpt_dir, CONFIG_MAP[r]['ckpt_subdir'])}")

    print("\n📁 Thư mục lưu trữ Logs & Metrics:")
    for r in sorted(args.rows):
        if r in CONFIG_MAP:
            print(f"   - Row {r}: {os.path.join(base_log_dir, CONFIG_MAP[r]['log_subdir'])}")
    print("=" * 75)


if __name__ == "__main__":
    main()
