# 🚀 CẨM NANG CHẠY TOÀN BỘ CẤU HÌNH D2SA TRÊN GOOGLE COLAB A100 (40GB VRAM) & LƯU GOOGLE DRIVE

> **Mục tiêu:** Chạy 1 lần duy nhất để huấn luyện và đánh giá toàn bộ **6 cấu hình Ablation Study** có sẵn trong code trên tập dữ liệu **MVTec D2SA** (Table V).  
> **Môi trường:** Google Colab Pro / Pro+ với GPU **NVIDIA A100 (40GB VRAM)**.  
> **Mức tiêu thụ VRAM:** Mô hình tiêu thụ ~12 – 13 GB VRAM (chỉ chiếm ~30% của 40GB VRAM), **hoàn toàn an toàn, tốc độ tối đa, loại bỏ 100% rủi ro tràn bộ nhớ (OOM)**.  
> **Cơ chế lưu trữ:** Toàn bộ Checkpoint (`.pth`), Log text (`.log`), Metrics (`.jsonl`), và Báo cáo JSON được **ghi trực tiếp và tức thì vào Google Drive** sau mỗi epoch (cơ chế atomic flush + fsync).

---

## 📊 1. BẢNG MAPPING 6 CẤU HÌNH CÓ SẴN TRÊN D2SA (TABLE V)

| Row | Cấu hình trong code | Input | Edge Mask | Cat Emb (60) | Spatial Attn | Loss Function | Số tham số | VRAM trên A100 (Batch 16) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/` | **4ch** (RGB+Vis) | ❌ | ❌ | ❌ | BCEWithLogits | 34,352,027 | ~11.5 GB |
| **2** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/` | **5ch** (RGB+Vis+Edge) | ✅ | ❌ | ❌ | BCEWithLogits | 34,353,563 | ~12.0 GB |
| **3** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding/` | **5ch** (RGB+Vis+Edge) | ✅ | ❌ | ❌ | Occ-Aware (5x) + Dice | 34,353,563 | ~12.2 GB |
| **4** | `scripts/train_d2sa.py` (**Cấu hình chính / Main Config**) | **5ch** (RGB+Vis+Edge) | ✅ | ✅ (60) | ❌ | Occ-Aware (5x) + Dice | **34,399,643** | ~12.8 GB |
| **5** | `other_config_d2sa/amodal_shape_prediction_no_embeding/` | **5ch** (RGB+Vis+Edge) | ✅ | ❌ | ✅ (k=7) | Occ-Aware (5x) + Dice | 34,353,661 | ~12.5 GB |
| **6** | `other_config_d2sa/amodal_shape_prediction_full_config/` | **5ch** (RGB+Vis+Edge) | ✅ | ✅ (60) | ✅ (k=7) | Occ-Aware (5x) + Dice | 34,399,741 | ~13.0 GB |

---

## ⚡ 2. CÁCH 1: DÙNG NOTEBOOK 1-CLICK (KHUYÊN DÙNG)

Trong repo đã có sẵn file Notebook Jupyter được thiết kế chuyên biệt cho Colab A100:  
👉 **[`train_d2sa_colab_a100.ipynb`](train_d2sa_colab_a100.ipynb)**

### Các bước thực hiện:
1. Mở [Google Colab](https://colab.research.google.com/).
2. Chọn **File** → **Upload notebook** → Chọn file `train_d2sa_colab_a100.ipynb`.
3. Đổi môi trường thực thi: **Runtime** → **Change runtime type** → Chọn **A100 GPU** và **High-RAM**.
4. Bấm **Runtime** → **Run all** (hoặc chạy từng cell theo thứ tự).

---

## 💻 3. CÁCH 2: CHẠY TRỰC TIẾP QUA CÁC CELL TRÊN COLAB

### Cell 1: Kết nối Google Drive & Kiểm tra GPU A100 (40GB)
```python
from google.colab import drive
import os
import torch

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Tạo sẵn thư mục lưu trữ trên Google Drive của bạn
DRIVE_DIR = "/content/drive/MyDrive/amodal_shape_prediction"
os.makedirs(f"{DRIVE_DIR}/checkpoints/d2sa", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/logs/d2sa", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/results/d2sa", exist_ok=True)

# 3. Kiểm tra GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"✅ Dữ liệu checkpoints/logs sẽ lưu tại: {DRIVE_DIR}")
```

### Cell 2: Chuẩn Bị Source Code (Tự nhận diện Drive hoặc Git) & Cài Đặt Thư Viện
```python
import os

DRIVE_REPO = "/content/drive/MyDrive/amodal_shape_prediction"
LOCAL_REPO = "/content/amodal_shape_prediction"

# Nếu đã upload thư mục amodal_shape_prediction lên Google Drive:
if os.path.exists(os.path.join(DRIVE_REPO, "scripts")):
    print("📁 Phát hiện repo có sẵn trên Google Drive! Sao chép sang SSD Colab để nạp code siêu nhanh...")
    !cp -r {DRIVE_REPO} {LOCAL_REPO}
    %cd {LOCAL_REPO}
# Nếu chưa có trên Drive -> tự động clone từ GitHub:
else:
    print("🌐 Chưa có repo trên Drive, đang clone từ GitHub...")
    !git clone https://github.com/vothenguyen/amodal_shape_prediction.git {LOCAL_REPO}
    %cd {LOCAL_REPO}
    !git checkout feat/d2sa

!pip install -q timm albumentations pycocotools opencv-python tqdm
```

### Cell 3: Tải Dữ Liệu D2SA Vào Ổ Cứng SSD Siêu Tốc Của Colab
*(Tải 4.01 GB mất ~1-2 phút với băng thông của Colab. Ảnh được giải nén vào ổ SSD local để nạp dữ liệu nhanh nhất, trong khi kết quả checkpoint/log sẽ lưu vào Google Drive).*

```bash
%%bash
mkdir -p data/D2SA
cd data/D2SA

# 1. Tải và giải nén Annotations (15.3 MB)
if [ ! -f "D2S_amodal_training_rot0.json" ]; then
    echo "📥 Đang tải Annotations D2SA..."
    curl -L -o "d2s_amodal_annotations_v1.tar.xz" \
        "https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938643-1629954673/d2s_amodal_annotations_v1.tar.xz" --progress-bar
    tar -xf d2s_amodal_annotations_v1.tar.xz
    rm -f d2s_amodal_annotations_v1.tar.xz
fi

# 2. Tải và giải nén Images (4.01 GB gồm 22,562 ảnh)
if [ ! -d "images" ] || [ $(ls -1 images 2>/dev/null | wc -l) -lt 20000 ]; then
    echo "📥 Đang tải Images D2SA (~4 GB)..."
    curl -L -o "d2s_amodal_images_v1.tar.xz" \
        -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36" \
        -H "Referer: https://www.mvtec.com/research-teaching/datasets/mvtec-d2s/downloads" \
        --connect-timeout 30 --max-time 7200 --progress-bar \
        "https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420939124-1629954676/d2s_amodal_images_v1.tar.xz"
    tar -xf d2s_amodal_images_v1.tar.xz
    rm -f d2s_amodal_images_v1.tar.xz
fi

echo "✅ Kiểm tra tính toàn vẹn dataset:"
cd ../..
python scripts/verify_d2sa_dataset.py
```

### Cell 4: 🔥 CHẠY 1 LẦN DUY NHẤT TOÀN BỘ 6 CẤU HÌNH (TABLE V)
```python
# Lệnh tự động chạy tuần tự từ Row 1 đến Row 6:
# - Mỗi cấu hình huấn luyện 30 epochs
# - Batch size = 16 (tối ưu tốc độ, tiêu thụ ~12.5GB / 40GB VRAM)
# - Num workers = 8 (tối ưu đọc ảnh đa luồng)
# - Đánh giá cuối kỳ trên toàn bộ 15,654 mẫu validation
# - Tự động ghi log, checkpoint, và JSON sang Google Drive

!python scripts/run_ablation_d2sa.py \
    --rows 1 2 3 4 5 6 \
    --mode all \
    --epochs 30 \
    --batch-size 16 \
    --accumulation-steps 1 \
    --num-workers 8 \
    --device cuda \
    --drive-dir /content/drive/MyDrive/amodal_shape_prediction
```

### Cell 5: Xem Bảng Tổng Hợp Table V Trực Tiếp Trên Colab
```python
from IPython.display import display, Markdown

table_path = "/content/drive/MyDrive/amodal_shape_prediction/results/d2sa/table_v_summary.md"
with open(table_path, "r", encoding="utf-8") as f:
    display(Markdown(f.read()))
```

---

## 🛡️ 4. KHẢ NĂNG PHỤC HỒI KHI BỊ NGẮT KẾT NỐI (RESUME)

Nếu Colab bị gián đoạn giữa chừng do hết thời gian phiên (timeout) hoặc mất mạng:
- Mọi checkpoint của từng epoch đã được ghi an toàn trên Google Drive (`swin_amodal_epoch_X.pth` và `last.pth`).
- Trạng thái của AdamW moments và learning rate Cosine đã được bảo toàn nguyên vẹn 100%.

Để tiếp tục huấn luyện một Row cụ thể từ epoch bị ngắt (ví dụ Row 4 tại epoch 15):
```python
!python scripts/run_ablation_d2sa.py \
    --rows 4 \
    --mode all \
    --epochs 30 \
    --batch-size 16 \
    --resume-epoch 15 \
    --device cuda \
    --drive-dir /content/drive/MyDrive/amodal_shape_prediction
```

---

## 📁 5. CẤU TRÚC KẾT QUẢ ĐÃ LƯU TRÊN GOOGLE DRIVE CỦA BẠN

Sau khi chạy xong, trong Google Drive tại `MyDrive/amodal_shape_prediction/` sẽ có đầy đủ:

```text
MyDrive/amodal_shape_prediction/
├── checkpoints/
│   └── d2sa/
│       ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/  (Row 1: epoch 1..30, last.pth)
│       ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/    (Row 2: epoch 1..30, last.pth)
│       ├── amodal-shape-prediction-no-spatial-no-embeding/                     (Row 3: epoch 1..30, last.pth)
│       ├── amodal_shape_prediction_main_config/                                (Row 4: epoch 1..30, last.pth)
│       ├── amodal_shape_prediction_no_embeding/                                (Row 5: epoch 1..30, last.pth)
│       └── amodal_shape_prediction_full_config/                                (Row 6: epoch 1..30, last.pth)
├── logs/
│   └── d2sa/
│       ├── row1_bce_vis_noedge_noemb_nospatial/ (*_train.log, *_metrics.jsonl)
│       ├── row2_bce_vis_edge_noemb_nospatial/   (*_train.log, *_metrics.jsonl)
│       ├── row3_occ_edge_noemb_nospatial/       (*_train.log, *_metrics.jsonl)
│       ├── row4_main_config/                    (*_train.log, *_metrics.jsonl)
│       ├── row5_occ_edge_noemb_spatial/         (*_train.log, *_metrics.jsonl)
│       └── row6_full_config/                    (*_train.log, *_metrics.jsonl)
└── results/
    └── d2sa/
        ├── row1_eval.json
        ├── row2_eval.json
        ├── row3_eval.json
        ├── row4_eval.json
        ├── row5_eval.json
        ├── row6_eval.json
        ├── table_v_summary.md                   # 🌟 Bảng tổng hợp so sánh Table V
        └── table_v_summary.json                 # 🌟 Dữ liệu JSON Table V để vẽ biểu đồ
```
