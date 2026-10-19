# 🚀 CẨM NANG CHẠY TOÀN BỘ CẤU HÌNH D2SA TRÊN GOOGLE COLAB A100 (40GB VRAM) & LƯU GOOGLE DRIVE

> **Mục tiêu:** Chạy 1 lần duy nhất để huấn luyện và đánh giá toàn bộ **6 cấu hình Ablation Study** có sẵn trong code trên tập dữ liệu **MVTec D2SA** (Table V).  
> **Môi trường:** Google Colab Pro / Pro+ với GPU **NVIDIA A100 (40GB VRAM)**.  
> **Mức tiêu thụ VRAM:** Mô hình tiêu thụ ~12 – 13 GB VRAM (chỉ chiếm ~30% của 40GB VRAM), **hoàn toàn an toàn, tốc độ tối đa, loại bỏ 100% rủi ro tràn bộ nhớ (OOM)**.  
> **Cơ chế lưu trữ:** Toàn bộ Checkpoint (`.pth`), Log text (`.log`), Metrics (`.jsonl`), và Báo cáo JSON được **ghi trực tiếp và tức thì vào Google Drive** sau mỗi epoch (cơ chế atomic flush + fsync).

---

## 📊 1. BẢNG MAPPING 6 CẤU HÌNH CÓ SẴN TRÊN D2SA (TABLE V)

|  Row  | Cấu hình trong code                                                                    |         Input          | Edge Mask | Cat Emb (60) | Spatial Attn |     Loss Function     |   Số tham số   | VRAM trên A100 (Batch 16) |
| :---: | :------------------------------------------------------------------------------------- | :--------------------: | :-------: | :----------: | :----------: | :-------------------: | :------------: | :-----------------------: |
| **1** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/` |   **4ch** (RGB+Vis)    |    ❌     |      ❌      |      ❌      |     BCEWithLogits     |   34,352,027   |         ~11.5 GB          |
| **2** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/`   | **5ch** (RGB+Vis+Edge) |    ✅     |      ❌      |      ❌      |     BCEWithLogits     |   34,353,563   |         ~12.0 GB          |
| **3** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding/`                    | **5ch** (RGB+Vis+Edge) |    ✅     |      ❌      |      ❌      | Occ-Aware (5x) + Dice |   34,353,563   |         ~12.2 GB          |
| **4** | `scripts/train_d2sa.py` (**Cấu hình chính / Main Config**)                             | **5ch** (RGB+Vis+Edge) |    ✅     |   ✅ (60)    |      ❌      | Occ-Aware (5x) + Dice | **34,399,643** |         ~12.8 GB          |
| **5** | `other_config_d2sa/amodal_shape_prediction_no_embeding/`                               | **5ch** (RGB+Vis+Edge) |    ✅     |      ❌      |   ✅ (k=7)   | Occ-Aware (5x) + Dice |   34,353,661   |         ~12.5 GB          |
| **6** | `other_config_d2sa/amodal_shape_prediction_full_config/`                               | **5ch** (RGB+Vis+Edge) |    ✅     |   ✅ (60)    |   ✅ (k=7)   | Occ-Aware (5x) + Dice |   34,399,741   |         ~13.0 GB          |

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
BRANCH = "feat/colab-a100"

# 1. Nếu đã upload thư mục amodal_shape_prediction lên Google Drive:
if os.path.exists(os.path.join(DRIVE_REPO, "scripts")):
    print("📁 Phát hiện repo có sẵn trên Google Drive! Sao chép sang SSD Colab để nạp code siêu nhanh...")
    os.makedirs(LOCAL_REPO, exist_ok=True)
    !cp -rn {DRIVE_REPO}/* {LOCAL_REPO}/ 2>/dev/null || cp -r {DRIVE_REPO}/* {LOCAL_REPO}/
    %cd {LOCAL_REPO}
# 2. Nếu repo cục bộ đã tồn tại (đã clone từ trước trên Colab):
elif os.path.exists(os.path.join(LOCAL_REPO, ".git")):
    print(f"🔄 Thư mục cục bộ đã tồn tại, đang fetch và chuyển sang branch {BRANCH}...")
    %cd {LOCAL_REPO}
    !git fetch origin
    !git checkout {BRANCH}
    !git pull origin {BRANCH}
# 3. Nếu chưa có trên Drive và chưa có cục bộ -> clone trực tiếp branch từ GitHub:
else:
    print(f"🌐 Chưa có repo, đang clone branch {BRANCH} từ GitHub...")
    !rm -rf {LOCAL_REPO}
    !git clone -b {BRANCH} https://github.com/vothenguyen/amodal_shape_prediction.git {LOCAL_REPO}
    %cd {LOCAL_REPO}

!pip install -q timm albumentations pycocotools opencv-python tqdm
```

### Cell 3: Thiết Lập & Tự Động Kết Nối Dữ Liệu D2SA

_(Tập dữ liệu D2SA gồm 3 file annotation `.json` (~15.3 MB) và thư mục `images/` gồm 22,562 file ảnh ~4 GB)._

> 💡 **Khuyên dùng:** Nếu bạn đã tải hoặc sao chép thư mục `D2SA` lên Google Drive (tại `/content/drive/MyDrive/amodal_shape_prediction/data/D2SA` hoặc `MyDrive/D2SA`), đoạn mã dưới đây sẽ tự động nhận diện và tạo liên kết tượng trưng (symlink) tức thì sang ổ SSD Colab!

```python
import os
import shutil

LOCAL_DATA_DIR = "/content/amodal_shape_prediction/data/D2SA"
os.makedirs("/content/amodal_shape_prediction/data", exist_ok=True)

# 1. Danh sách các vị trí ứng viên trên Google Drive
CANDIDATES = [
    "/content/drive/MyDrive/amodal_shape_prediction/data/D2SA",
    "/content/drive/MyDrive/D2SA",
    "/content/drive/MyDrive/data/D2SA",
    "/content/data/D2SA"
]

req_file = "D2S_amodal_training_rot0.json"
found_drive_path = None

for cand in CANDIDATES:
    if os.path.exists(os.path.join(cand, req_file)) and os.path.isdir(os.path.join(cand, "images")):
        found_drive_path = cand
        break

if found_drive_path:
    print(f"📁 Tìm thấy bộ dữ liệu D2SA đầy đủ trên Google Drive tại: {found_drive_path}")
    if os.path.exists(LOCAL_DATA_DIR) and os.path.islink(LOCAL_DATA_DIR):
        os.unlink(LOCAL_DATA_DIR)
    elif os.path.exists(LOCAL_DATA_DIR) and not os.listdir(LOCAL_DATA_DIR):
        os.rmdir(LOCAL_DATA_DIR)
    
    if not os.path.exists(LOCAL_DATA_DIR):
        print(f"🔗 Tạo symbolic link từ Google Drive sang SSD Colab: {LOCAL_DATA_DIR}...")
        os.symlink(found_drive_path, LOCAL_DATA_DIR, target_is_directory=True)
        print("✅ Kết nối dữ liệu thành công qua symlink!")
    else:
        print("✅ Thư mục cục bộ đã sẵn sàng dữ liệu.")
else:
    # Kiểm tra xem có file nén .zip trên Drive không để giải nén tự động
    zip_candidates = [
        "/content/drive/MyDrive/amodal_shape_prediction/data/D2SA.zip",
        "/content/drive/MyDrive/D2SA.zip",
        "/content/drive/MyDrive/d2s_amodal.zip",
    ]
    unzipped = False
    for zpath in zip_candidates:
        if os.path.exists(zpath):
            print(f"📦 Phát hiện file nén trên Drive: {zpath}. Đang giải nén vào {LOCAL_DATA_DIR}...")
            os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
            !unzip -q "{zpath}" -d "{LOCAL_DATA_DIR}"
            unzipped = True
            break
            
    if not unzipped:
        print("⚠️ Chưa phát hiện thư mục D2SA trên Google Drive!")
        print("💡 HƯỚNG DẪN ĐƯA DỮ LIỆU LÊN COLAB:")
        print("  Cách 1 (Khuyên dùng): Tải thư mục 'D2SA' (chứa images/ và 3 file JSON) lên Google Drive tại:")
        print("         /content/drive/MyDrive/amodal_shape_prediction/data/D2SA/")
        print("  Cách 2: Giải nén hoặc kéo thả trực tiếp thư mục D2SA vào ổ SSD Colab tại:")
        print("         /content/amodal_shape_prediction/data/D2SA/")

# Kiểm tra tính toàn vẹn
!python scripts/verify_d2sa_dataset.py
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
