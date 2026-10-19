# 🔬 THỰC NGHIỆM ABLATION STUDY D2SA (NVIDIA A100 40GB/80GB VRAM)

> 📖 Hướng dẫn chi tiết chạy 1 lần toàn bộ 6 cấu hình trên Google Colab A100 và lưu trực tiếp vào Google Drive xem tại:  
> 👉 **[COLAB_A100_GUIDE.md](../COLAB_A100_GUIDE.md)**  
> 👉 File Notebook sẵn sàng chạy 1-click: **[train_d2sa_colab_a100.ipynb](../train_d2sa_colab_a100.ipynb)**

---

## 🗂️ Cấu Trúc Thư Mục `scripts/`

```text
scripts/
├── model.py                         # Kiến trúc Swin-UNet (5 kênh, Category Embedding 60 lớp)
├── dataset_d2sa.py                  # Dataset class cho D2SA (đọc RLE, 60 lớp, bounding box pad)
├── train_d2sa.py                    # Script huấn luyện Row 4 cho D2SA (Main config chuẩn)
├── evaluate_d2sa.py                 # Đánh giá D2SA (mIoU, Dice, Precision, Recall, Inv-mIoU)
├── verify_d2sa_dataset.py           # Script kiểm tra tính toàn vẹn dataset trước khi train
├── logging_utils.py                 # Hệ thống log chuẩn: Immediate Flush, fsync, SIGTERM handling
├── run_ablation_d2sa.py             # Runner Python tự động chạy toàn bộ hoặc từng cấu hình
├── run_ablation_d2sa.sh             # Runner Bash tối ưu cho GPU A100 / Linux / Colab
├── other_config_d2sa/               # 5 cấu hình ablation còn lại của Table V
│   ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/  # Row 1
│   ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/    # Row 2
│   ├── amodal-shape-prediction-no-spatial-no-embeding/                     # Row 3
│   ├── amodal_shape_prediction_no_embeding/                                # Row 5
│   └── amodal_shape_prediction_full_config/                                # Row 6
├── other_config_cocoa/              # Các cấu hình thực nghiệm trên tập COCOA
├── dataset.py                       # Dataset class cho COCOA
├── train.py                         # Huấn luyện trên COCOA
└── evaluate.py                      # Đánh giá trên COCOA
```

---

## 📊 Bảng 6 Cấu Hình Ablation Study (Table V)

| Row | Cấu hình | Kênh vào (Input) | Edge Mask | Cat Embedding (60) | Spatial Attention | Hàm mất mát (Loss) | Số tham số | VRAM trên A100 (Batch 16) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `other_config_d2sa/...noedge-bce` | **4** (RGB + Vis) | ❌ | ❌ | ❌ | BCEWithLogits | 34,352,027 | ~11.5 GB |
| **2** | `other_config_d2sa/...edge-bce` | **5** (RGB + Vis + Edge) | ✅ | ❌ | ❌ | BCEWithLogits | 34,353,563 | ~12.0 GB |
| **3** | `other_config_d2sa/...no-embeding` | **5** (RGB + Vis + Edge) | ✅ | ❌ | ❌ | Occ-Aware (5x) + Dice | 34,353,563 | ~12.2 GB |
| **4** | `scripts/train_d2sa.py` (**Main**) | **5** (RGB + Vis + Edge) | ✅ | ✅ (60) | ❌ | Occ-Aware (5x) + Dice | 34,399,643 | ~12.8 GB |
| **5** | `other_config_d2sa/...no_embeding` | **5** (RGB + Vis + Edge) | ✅ | ❌ | ✅ (k=7) | Occ-Aware (5x) + Dice | 34,353,661 | ~12.5 GB |
| **6** | `other_config_d2sa/...full_config` | **5** (RGB + Vis + Edge) | ✅ | ✅ (60) | ✅ (k=7) | Occ-Aware (5x) + Dice | 34,399,741 | ~13.0 GB |

---

## ⚡ Lệnh Chạy Nhanh Cho NVIDIA A100 (40GB/80GB VRAM)

```bash
# 1. Kiểm tra tính toàn vẹn dataset D2SA:
python scripts/verify_d2sa_dataset.py

# 2. Chạy 1 lần đầy đủ 6 cấu hình (Batch 16, 30 epochs) và lưu thẳng vào Google Drive:
python scripts/run_ablation_d2sa.py \
    --mode all \
    --epochs 30 \
    --batch-size 16 \
    --accumulation-steps 1 \
    --num-workers 8 \
    --drive-dir /content/drive/MyDrive/amodal_shape_prediction
```
