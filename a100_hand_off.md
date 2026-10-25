# 🚀 CẨM NANG BÀN GIAO CHẠY THỰC NGHIỆM KLTN TRÊN COLAB PRO A100
# (TỐI ƯU HÓA: CHỈ CHẠY KIẾN TRÚC MỚI ROW 7 & ĐỐI CHỨNG D2SA)

> **Dự án:** Khóa luận Tốt nghiệp — Dự đoán biên dạng hoàn chỉnh của vật thể bị che khuất (**Amodal Shape Prediction**)  
> **Người thực hiện:** Võ Thế Nguyễn (Ryan) & Phan Quang Nghĩa  
> **Môi trường:** Google Colab Pro — **GPU A100 (200 Compute Units)**  

---

## 📌 1. TẠI SAO CHỈ CẦN CHẠY CÁC CẤU HÌNH DƯỚI ĐÂY?

1. **Trên COCOA và KINS:** Nhóm đã chạy xong toàn bộ các cấu hình cũ (Row 1 -> Row 6) từ đợt công bố bài báo **MAPR 2026** và đã có sẵn bảng điểm chuẩn trong bài báo:
   - **KINS (Row 4 - MAPR Base):** `88.34%` Overall mIoU | `57.36%` Occ. mIoU.
   - **COCOA (Row 4 - MAPR Base):** `84.69%` Overall mIoU | `53.87%` Occ. mIoU (và Row 3 No-Embedding là `57.49%` Occ. mIoU).
   👉 **Do đó, trên COCOA và KINS KHÔNG CẦN train lại bất kỳ cấu hình cũ nào!** Chỉ cần chạy duy nhất **Row 7 (Kiến trúc cải tiến mới của KLTN: `SG-GBA Bottleneck`)** để đọ điểm trực tiếp với bài báo MAPR.

2. **Trên D2SA (Dataset mới mở rộng cho KLTN):**
   - Trước đó nhóm mới viết xong code mở rộng sang D2SA và chạy Smoke Test 1 epoch trên máy local chứ chưa train 30 epochs trên A100.
   - Để tiết kiệm tối đa Compute Units mà vẫn có bảng so sánh trực diện giữa **Bài báo MAPR (Row 4)** và **Khóa luận tốt nghiệp (Row 7)** trên D2SA, chỉ cần ưu tiên chạy **Row 7 (SG-GBA)** và **Row 4 (MAPR Base)** (kèm thêm Row 6 nếu muốn vẽ thêm cột Attention Paradox). Không cần chạy Row 1, 2, 3, 5 tốn thời gian!

---

## 📌 2. BẢNG PHÂN BỔ NHIỆM VỤ SIÊU GỌN TRÊN A100 (~45 CUs TỔNG CỘNG)

| Thứ tự | Tập dữ liệu | Cấu hình cần chạy (30 Epochs) | Thời gian A100 | Số CUs dự kiến | Mục đích trong quyển KLTN & Web Demo |
|:---:|:---:|:---|:---:|:---:|:---|
| **1** | **COCOA** | Chỉ chạy **Row 7 (`SG-GBA`)** | ~25 phút | ~6 CUs | Chứng minh FiLM + Bottleneck Attention khắc phục lỗi tụt điểm của Category Embedding trên COCOA ở bài MAPR. |
| **2** | **D2SA** | Chạy **Row 7 (`SG-GBA`)** & **Row 4 (`MAPR Base`)** | ~1.5 giờ (cả 2) | ~20 CUs | Lấy cặp số liệu & cặp checkpoint `.pth` đối chứng (MAPR vs. KLTN) trên tập dữ liệu mới D2SA để đưa vào báo cáo và `app.py`. |
| **3** | **KINS** | Chỉ chạy **Row 7 (`SG-GBA`)** | ~1.5 giờ | ~20 CUs | Chứng minh Row 7 vượt qua mốc `88.34% / 57.36%` của bài báo MAPR trên tập giao thông KINS. |

---

## 📌 3. CÁC CELL CHẠY TRÊN GOOGLE COLAB (COPY-PASTE)

Mở Google Colab -> Chọn **Runtime -> Change runtime type -> A100 GPU**, sau đó chạy:

### 🔹 CELL 1: Mount Google Drive, Clone Repo & Nối Symlink Lưu Trữ
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf amodal_shape_prediction
!git clone [https://github.com/vothenguyen/amodal_shape_prediction.git](https://github.com/vothenguyen/amodal_shape_prediction.git)
%cd /content/amodal_shape_prediction

# Tạo thư mục trên Drive để lưu bền vững checkpoints, logs, results
!mkdir -p /content/drive/MyDrive/KLTN_Amodal_2026/checkpoints
!mkdir -p /content/drive/MyDrive/KLTN_Amodal_2026/logs
!mkdir -p /content/drive/MyDrive/KLTN_Amodal_2026/results

!rm -rf checkpoints logs results
!ln -s /content/drive/MyDrive/KLTN_Amodal_2026/checkpoints checkpoints
!ln -s /content/drive/MyDrive/KLTN_Amodal_2026/logs logs
!ln -s /content/drive/MyDrive/KLTN_Amodal_2026/results results

!pip install -q timm albumentations pycocotools tqdm gradio ultralytics
print("✅ Môi trường A100 đã sẵn sàng!")
```

---

### 🔹 CELL 2: Chạy DUY NHẤT Row 7 (SG-GBA) trên tập COCOA (~25 phút)
```python
import os
%cd /content/amodal_shape_prediction

# 1. Giải nén annotations COCOA từ Drive và tải ảnh COCO2014 vào SSD Colab
!mkdir -p data
# Chỉnh lại đường dẫn annotations.zip trên Drive nếu đặt ở thư mục khác:
!unzip -qo /content/drive/MyDrive/CV+/data/annotations.zip -d data/

%cd /content/amodal_shape_prediction/data
if not os.path.exists("train2014"):
    !wget -q [http://images.cocodataset.org/zips/train2014.zip](http://images.cocodataset.org/zips/train2014.zip) && unzip -q train2014.zip && rm train2014.zip
if not os.path.exists("val2014"):
    !wget -q [http://images.cocodataset.org/zips/val2014.zip](http://images.cocodataset.org/zips/val2014.zip) && unzip -q val2014.zip && rm val2014.zip

!mkdir -p /content/data
!ln -sf /content/amodal_shape_prediction/data/train2014 /content/data/train2014
!ln -sf /content/amodal_shape_prediction/data/val2014 /content/data/val2014
!ln -sf /content/amodal_shape_prediction/data/annotations /content/data/annotations

# 2. Train & Evaluate duy nhất cấu hình mới Row 7 (EXPECTED_PARAMS = 34,867,566)
%cd /content/amodal_shape_prediction/scripts/other_config_cocoa/amodal_shape_prediction_sg_gba_bottleneck
!python train.py
!python evaluate.py
```

---

### 🔹 CELL 3: Chạy Row 7 (Mới) & Row 4 (MAPR Base) trên tập D2SA (~1.5 giờ)
```python
import os
%cd /content/amodal_shape_prediction

# 1. Giải nén D2SA từ Drive vào ổ cứng SSD của Colab (/content) để đọc I/O nhanh
!mkdir -p /content/amodal_shape_prediction/data/D2SA
D2SA_ZIP = "/content/drive/MyDrive/KLTN_Amodal_2026/D2SA.zip"  # Sửa lại đúng đường dẫn zip trên Drive
if os.path.exists(D2SA_ZIP):
    !unzip -qo "{D2SA_ZIP}" -d /content/amodal_shape_prediction/data/D2SA

# Kiểm tra dữ liệu D2SA
!python scripts/verify_d2sa_dataset.py

# 2. Chỉ chạy Row 7 (Kiến trúc mới KLTN) và Row 4 (Kiến trúc chuẩn MAPR để đối chứng):
!python scripts/run_ablation_d2sa.py --rows 7 4 --epochs 30 --batch-size 64 --num-workers 8
```
*(Ghi chú: Nếu giữa chừng bị văng Colab ở epoch `N`, chỉ cần thêm cờ `--resume-epoch N` vào cuối lệnh trên để chạy tiếp).*

---

### 🔹 CELL 4: Chạy DUY NHẤT Row 7 (SG-GBA) trên tập KINS (~1.5 giờ)
```python
import os, kagglehub
%cd /content/amodal_shape_prediction

kitti_path = kagglehub.dataset_download("anupammajhi/kitti-2d-object-detection")
kins_path = kagglehub.dataset_download("ryanthenguyen/kins-data")

!rm -rf /content/kitti_data /content/kins_data
os.symlink(f"{kitti_path}/data_object_image_2", "/content/kitti_data")
os.symlink(kins_path, "/content/kins_data")
print("✅ Đã nối xong dữ liệu KINS! Hãy mở phần cuối của scripts/train_kins.ipynb để chạy Cell của Row 7 (EXPECTED_PARAMS = 34,813,038).")
```

---

## 📌 4. FILE CẦN THU HOẠCH GỬI LẠI CHO RYAN
Sau khi chạy xong 3 Cell trên, toàn bộ kết quả đã nằm gọn trong `/content/drive/MyDrive/KLTN_Amodal_2026/`:
1. **Checkpoints `.pth` ở Epoch 30** của:
   - `COCOA Row 7`
   - `D2SA Row 7` và `D2SA Row 4`
   - `KINS Row 7`
2. **File JSON kết quả đánh giá (`results/*.json`)** và **File Log (`logs/*.jsonl`)** tương ứng.
3. Nh
