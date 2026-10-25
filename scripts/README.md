# 🔬 THỰC NGHIỆM ABLATION STUDY D2SA (NVIDIA H200 141GB VRAM)

> 📖 **TÀI LIỆU HỢP NHẤT CHÍNH THỨC:**  
> Toàn bộ hướng dẫn thực nghiệm Table V, đối chiếu lý thuyết toán học 6 cấu hình, cơ chế checkpoint 2 tầng ghi nguyên tử (Atomic write), quy chuẩn khôi phục learning rate Cosine khi resume, và cẩm nang vận hành trên GPU **NVIDIA H200 (141GB VRAM)** đã được hợp nhất thành một tài liệu duy nhất tại gốc dự án:  
> 
> 👉 **[h200_training_handoff_guide.md](../h200_training_handoff_guide.md)**

---

## 🗂️ Cấu Trúc Thư Mục `scripts/`

```text
scripts/
├── model.py                         # Kiến trúc Swin-UNet chuẩn (Row 4 - Main Config)
├── dataset.py                       # Dataset class cho COCOA (đọc polygon & sequence order)
├── dataset_d2sa.py                  # Dataset class cho D2SA (đọc RLE, 60 lớp, bounding box pad)
├── train.py                         # Script huấn luyện Row 4 cho COCOA
├── train_d2sa.py                    # Script huấn luyện Row 4 cho D2SA (Main config chuẩn)
├── evaluate.py                      # Đánh giá COCOA (có cờ --num-classes)
├── evaluate_d2sa.py                 # Đánh giá D2SA (mIoU, Dice, Precision, Recall, Inv-mIoU)
├── verify_d2sa_dataset.py           # Script kiểm tra nhanh tính toàn vẹn dataset trước khi train
├── logging_utils.py                 # Hệ thống log chuẩn: Immediate Flush, fsync, SIGTERM handling
├── run_ablation_d2sa.py             # Runner Python tự động chạy toàn bộ hoặc từng cấu hình
├── run_ablation_d2sa.sh             # Runner Bash tối ưu cho cụm máy chủ H200 / Linux / Slurm
└── other_config_d2sa/               # 5 cấu hình ablation còn lại của Table V
    ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/  # Row 1
    ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/    # Row 2
    ├── amodal-shape-prediction-no-spatial-no-embeding/                     # Row 3
    ├── amodal_shape_prediction_no_embeding/                                # Row 5
    └── amodal_shape_prediction_full_config/                                # Row 6
```

---

## ⚡ Lệnh Chạy Nhanh Cho H200 (141GB VRAM)

```bash
# 1. Kiểm tra tính toàn vẹn dataset:
python scripts/verify_d2sa_dataset.py

# 2. Smoke test 1 epoch trên H200:
python scripts/train_d2sa.py --epochs 1 --subset-size 50 --batch-size 4 --accumulation-steps 1 --num-workers 4 --device cuda

# 3. Chạy full 6 cấu hình (Batch 16, Accumulation 1 -> Effective Batch 16, 8 Workers):
bash scripts/run_ablation_d2sa.sh cuda 30 16 1 8

# 4. Chạy nền qua nohup:
nohup bash scripts/run_ablation_d2sa.sh cuda 30 16 1 8 > ablation_runner.log 2>&1 &
echo $! > ablation_runner.pid

# 5. Theo dõi tiến trình thời gian thực:
tail -f logs/d2sa/row4_main_config/*_train.log
```

Chi tiết toàn văn hướng dẫn xem tại: [`../h200_training_handoff_guide.md`](../h200_training_handoff_guide.md).
