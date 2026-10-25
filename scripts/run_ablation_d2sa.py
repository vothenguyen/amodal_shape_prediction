"""
===================================================================================
RUN ABLATION D2SA - Chạy toàn bộ hoặc từng cấu hình ablation trên D2SA
===================================================================================
Script tự động hóa huấn luyện và đánh giá 6 cấu hình ablation trên D2SA:
- Row 1: 4ch (RGB+Vis) | No Edge | No Emb | No Spatial | BCE Loss
- Row 2: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | BCE Loss
- Row 3: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | Occ-Aware Loss (5x)
- Row 4: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | No Spatial | Occ-Aware Loss (5x) [Main]
- Row 5: 5ch (RGB+Vis+Edge) | Edge | No Emb | Spatial Attn | Occ-Aware Loss (5x)
- Row 6: 5ch (RGB+Vis+Edge) | Edge | Emb (60) | Spatial Attn | Occ-Aware Loss (5x) [Full]

Ví dụ sử dụng:
  # Chạy thử smoke test 6 config trên subset 30 mẫu, 1 epoch:
  python scripts/run_ablation_d2sa.py --mode train --epochs 1 --subset-size 30 --batch-size 2

  # Chạy full huấn luyện Row 4 trên GPU:
  python scripts/run_ablation_d2sa.py --rows 4 --mode train --epochs 30 --device cuda

  # Đánh giá toàn bộ 6 config:
  python scripts/run_ablation_d2sa.py --mode eval --device cuda
===================================================================================
"""

import os
import sys
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
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce"),
        "log_dir": os.path.join(ROOT_DIR, "logs", "d2sa", "row1_bce_vis_noedge_noemb_nospatial"),
    },
    2: {
        "name": "Row 2 (5ch, BCE, Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "evaluate.py"),
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "d2sa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce"),
        "log_dir": os.path.join(ROOT_DIR, "logs", "d2sa", "row2_bce_vis_edge_noemb_nospatial"),
    },
    3: {
        "name": "Row 3 (5ch, OccLoss 5x, Edge, No Emb, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal-shape-prediction-no-spatial-no-embeding", "evaluate.py"),
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "d2sa", "amodal-shape-prediction-no-spatial-no-embeding"),
        "log_dir": os.path.join(ROOT_DIR, "logs", "d2sa", "row3_occ_edge_noemb_nospatial"),
    },
    4: {
        "name": "Row 4 (Main Config: 5ch, OccLoss 5x, Edge, Emb 60, No Spatial)",
        "train_script": os.path.join(SCRIPTS_DIR, "train_d2sa.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "evaluate_d2sa.py"),
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "d2sa", "amodal_shape_prediction_main_config"),
        "log_dir": os.path.join(ROOT_DIR, "logs", "d2sa", "row4_main_config"),
    },
    5: {
        "name": "Row 5 (5ch, OccLoss 5x, Edge, No Emb, Spatial Attn)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_no_embeding", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_no_embeding", "evaluate.py"),
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "d2sa", "amodal_shape_prediction_no_embeding"),
        "log_dir": os.path.join(ROOT_DIR, "logs", "d2sa", "row5_occ_edge_noemb_spatial"),
    },
    6: {
        "name": "Row 6 (Full Config: 5ch, OccLoss 5x, Edge, Emb 60, Spatial Attn)",
        "train_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_full_config", "train.py"),
        "eval_script": os.path.join(SCRIPTS_DIR, "other_config_d2sa", "amodal_shape_prediction_full_config", "evaluate.py"),
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "d2sa", "amodal_shape_prediction_full_config"),
        "log_dir": os.path.join(ROOT_DIR, "logs", "d2sa", "row6_full_config"),
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


def main():
    parser = argparse.ArgumentParser(description="Ablation Runner cho D2SA (6 cấu hình)")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="Danh sách Row cần chạy (1–6)")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "all"], default="all", help="Chế độ: train, eval, hoặc all")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (Mặc định 16 tối ưu cho NVIDIA H200 141GB)")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps (Mặc định 1 cho H200)")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu huấn luyện (0: toàn bộ)")
    parser.add_argument("--val-subset-size", type=int, default=0, help="Số mẫu validation/đánh giá (0: toàn bộ)")
    parser.add_argument("--num-workers", type=int, default=8, help="Số workers cho DataLoader (Mặc định 8 cho máy chủ H200)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Epoch tiếp tục huấn luyện (0: từ đầu)")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Đường dẫn trực tiếp tới checkpoint cần resume")
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "=" * 75)
    print("🏆 RUNNER ABLATION STUDY D2SA (MVTec D2S Amodal)")
    print(f"   Rows cần chạy: {args.rows}")
    print(f"   Chế độ:        {args.mode}")
    print(f"   Số epoch:      {args.epochs}")
    print(f"   Batch size:    {args.batch_size} (Tích lũy: {args.accumulation_steps})")
    print(f"   Train subset:  {args.subset_size if args.subset_size > 0 else 'Full dataset'}")
    print(f"   Val subset:    {args.val_subset_size if args.val_subset_size > 0 else 'Full dataset'}")
    print(f"   Num workers:   {args.num_workers}")
    print("=" * 75)

    success_rows = []
    failed_rows = []

    for r in sorted(args.rows):
        if r not in CONFIG_MAP:
            print(f"⚠️ Row {r} không hợp lệ (chỉ hỗ trợ 1..6)")
            continue

        cfg = CONFIG_MAP[r]
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
            ckpt_path = os.path.join(cfg["ckpt_dir"], f"swin_amodal_epoch_{args.epochs}.pth")
            eval_cmd = [
                python_exe, cfg["eval_script"],
                "--checkpoint", ckpt_path,
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

    print("\n" + "=" * 75)
    print("📋 TỔNG KẾT TIẾN TRÌNH ABLATION D2SA")
    print("=" * 75)
    print(f"Thành công ({len(success_rows)}/{len(args.rows)}): {success_rows}")
    if failed_rows:
        print(f"Thất bại ({len(failed_rows)}/{len(args.rows)}):   {failed_rows}")
    print("\n📁 Thư mục lưu trữ Logs & Metrics:")
    for r in sorted(args.rows):
        if r in CONFIG_MAP:
            print(f"   - Row {r}: {CONFIG_MAP[r]['log_dir']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
