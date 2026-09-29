# 🔬 HƯỚNG DẪN THỰC NGHIỆM ABLATION STUDY (COCOA, KINS, D2SA)

Tài liệu này cung cấp bảng mapping chi tiết 6 cấu hình nghiên cứu thực nghiệm (Ablation Study) dựa trên đối chiếu 100% với mã nguồn thực tế, cùng hướng dẫn chạy huấn luyện và đánh giá trên tập dữ liệu **D2SA (MVTec D2S Amodal)** dành cho môi trường máy chủ GPU mạnh (**NVIDIA A100 40GB/80GB** hoặc cụm Slurm/Linux).

---

## 📊 1. Bảng Mapping 6 Cấu hình Thực nghiệm (Table V)

Bảng đối chiếu áp dụng chung cho cả 3 dataset: **COCOA**, **KINS**, và **D2SA**:

| Row | Tên cấu hình / Thư mục | Kênh vào (Input) | Edge Mask | Cat Embedding | Spatial Attention | Hàm mất mát (Loss) | Số tham số | Chữ ký `forward()` |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/` | **4** (RGB + Vis) | ❌ Tắt | ❌ Tắt | ❌ Tắt | BCEWithLogitsLoss | **34,352,027** | `forward(x)` |
| **2** | `amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/` | **5** (RGB + Vis + Edge) | ✅ Bật | ❌ Tắt | ❌ Tắt | BCEWithLogitsLoss | **34,353,563** | `forward(x)` |
| **3** | `amodal-shape-prediction-no-spatial-no-embeding/` | **5** (RGB + Vis + Edge) | ✅ Bật | ❌ Tắt | ❌ Tắt | Occlusion-Aware (5x) + Dice | **34,353,563** | `forward(x)` |
| **4** | **Cấu hình chính (Main / Baseline)**<br>• COCOA: `scripts/train.py`<br>• D2SA: `scripts/train_d2sa.py` | **5** (RGB + Vis + Edge) | ✅ Bật | ✅ **Bật**<br>(COCO: 91, D2SA: 60) | ❌ Tắt | Occlusion-Aware (5x) + Dice | **34,399,643** | `forward(x, class_ids)` |
| **5** | `amodal_shape_prediction_no_embeding/` | **5** (RGB + Vis + Edge) | ✅ Bật | ❌ Tắt | ✅ **Bật** (k=7) | Occlusion-Aware (5x) + Dice | **34,353,661** | `forward(x)` |
| **6** | `amodal_shape_prediction_full_config/` | **5** (RGB + Vis + Edge) | ✅ Bật | ✅ **Bật**<br>(COCO: 91, D2SA: 60) | ✅ **Bật** (k=7) | Occlusion-Aware (5x) + Dice | **34,399,741** | `forward(x, class_ids)` |

> **Ghi chú về quan hệ toán học giữa các cấu hình (Delta đối chiếu chính xác 100%):**
> - **Row 2 so với Row 1 (+1,536 tham số):** Thêm 1 kênh vào Patch Embedding của Swin (`patch_size=4×4`, `embed_dim=96`): $96 \times 1 \times 4 \times 4 = 1,536$. Tổng: $34,352,027 + 1,536 = 34,353,563$.
> - **Row 3 so với Row 2 (0 tham số):** Cùng kiến trúc mô hình, chỉ thay hàm mất mát từ BCE sang Occlusion-Aware (5x) + Dice. Tổng: $34,353,563$.
> - **Row 4 so với Row 3 (+46,080 tham số):** Thêm Category Embedding cho 60 lớp đối tượng vào bottleneck 768 chiều: $60 \times 768 = 46,080$. Tổng: $34,353,563 + 46,080 = 34,399,643$.
> - **Row 5 so với Row 3 (+98 tham số):** Thêm khối Spatial Attention (Conv2d kernel 7×7, 2 kênh vào [AvgPool, MaxPool], 1 kênh ra, `bias=False`): $1 \times 2 \times 7 \times 7 = 98$. Tổng: $34,353,563 + 98 = 34,353,661$.
> - **Row 6 so với Row 4 (+98 tham số):** Thêm khối Spatial Attention (98 tham số) trên nền Row 4 đã có Category Embedding. Tổng: $34,399,643 + 98 = 34,399,741$.
> - **Tất cả 6 cấu hình đều tích hợp sẵn assertion `assert total_params == EXPECTED_PARAMS` ngay khi khởi động**, khớp 100% với số lượng tham số ghi trong log xuất phát.

---

## 🗂️ 2. Cấu trúc Thư mục Mã nguồn

```text
scripts/
├── model.py                      # Kiến trúc Swin-UNet chuẩn (Row 4)
├── dataset.py                    # Dataset class cho COCOA (đọc polygon & order)
├── dataset_d2sa.py               # Dataset class cho D2SA (đọc RLE, 60 lớp)
├── train.py                      # Train Row 4 cho COCOA
├── train_d2sa.py                 # Train Row 4 cho D2SA (Main config)
├── evaluate.py                   # Đánh giá COCOA (có cờ --num-classes)
├── evaluate_d2sa.py              # Đánh giá D2SA (mIoU, Dice, Precision, Recall, Inv-mIoU)
├── verify_d2sa_dataset.py        # Script kiểm tra nhanh tính toàn vẹn dataset trước khi train
├── logging_utils.py              # Hệ thống log chuẩn: Immediate Flush, fsync, SIGTERM handling
├── run_ablation_d2sa.py          # Runner Python tự động chạy toàn bộ hoặc từng cấu hình
├── run_ablation_d2sa.sh          # Runner Bash chạy trên Linux / A100 / Slurm / Colab
└── other_config_d2sa/            # 5 cấu hình ablation D2SA
    ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/  # Row 1
    ├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/    # Row 2
    ├── amodal-shape-prediction-no-spatial-no-embeding/                     # Row 3
    ├── amodal_shape_prediction_no_embeding/                                # Row 5
    └── amodal_shape_prediction_full_config/                                # Row 6
```

---

## 🚀 3. Hướng dẫn Chạy Huấn luyện & Đánh giá trên D2SA

### 3.1. Chuẩn bị Môi trường & Tải Dữ liệu D2SA (Bắt buộc)

Do thư mục `data/` được loại trừ trong `.gitignore`, sau khi clone repository về máy chủ A100, cần tải và giải nén dữ liệu D2SA theo các bước dưới đây:

#### 1. Môi trường Python & Thư viện (Python 3.10 - 3.12, PyTorch 2.x):
```bash
pip install torch torchvision timm albumentations opencv-python tqdm pycocotools numpy matplotlib
```

#### 2. Tải Dữ liệu từ Trang chủ MVTec (Link Trực tiếp Chính thức):
Server `mydrive.ch` yêu cầu xử lý HTTP Redirect (`-L`) và có thể chặn các request tự động dạng bot nếu thiếu header trình duyệt. Dưới đây là 2 link gốc trực tiếp từ trang MVTec và lệnh curl kèm header giả lập trình duyệt **đã thực tế tải thành công 4.01 GB tại máy local ở Phase 3**:

```bash
# Tạo thư mục data/D2SA
mkdir -p data/D2SA && cd data/D2SA

# 1. Tải Annotations (~15.3 MB nén tar.xz, giải nén ~100 MB gồm 3 file JSON):
curl -L -o "d2s_amodal_annotations_v1.tar.xz" \
  "https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938643-1629954673/d2s_amodal_annotations_v1.tar.xz" --progress-bar

# 2. Tải Hình ảnh (~4.01 GB nén tar.xz, gồm 22,562 ảnh 1440x1920)
# (Lệnh thực tế đã tải thành công tại máy local ở Phase 3 kèm header giả lập trình duyệt để tránh bị mydrive.ch trả 404):
curl -L -o "d2s_amodal_images_v1.tar.xz" \
  -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36" \
  -H "Referer: https://www.mvtec.com/research-teaching/datasets/mvtec-d2s/downloads" \
  --connect-timeout 30 --max-time 7200 --progress-bar \
  "https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420939124-1629954676/d2s_amodal_images_v1.tar.xz"

# 3. Giải nén dữ liệu:
tar -xf d2s_amodal_annotations_v1.tar.xz
tar -xf d2s_amodal_images_v1.tar.xz

# Quay lại thư mục gốc dự án
cd ../..
```

> 🛡️ **Phương án dự phòng (Khuyến nghị nếu mạng server trường chặn kết nối ra ngoài):**
> Thầy có thể tải trước 2 file `d2s_amodal_annotations_v1.tar.xz` và `d2s_amodal_images_v1.tar.xz` bằng trình duyệt web (Chrome/Firefox) trên máy cá nhân qua link trực tiếp trên, sau đó chuyển file lên server qua Google Drive hoặc lệnh `scp`:
> ```bash
> scp d2s_amodal_images_v1.tar.xz d2s_amodal_annotations_v1.tar.xz user@server:/path/to/project/data/D2SA/
> ```

#### 3. Cấu trúc thư mục dữ liệu `data/D2SA/` sau khi giải nén:
```text
data/
└── D2SA/
    ├── images/                          # 22,562 ảnh JPG (1440x1920)
    │   ├── D2S_000000.jpg
    │   ├── D2S_000001.jpg
    │   └── ...
    ├── D2S_amodal_training_rot0.json    # 438 ảnh, 690 nhãn (train gốc)
    ├── D2S_amodal_augmented.json        # 1,562 ảnh, 12,376 nhãn (train tăng cường)
    └── D2S_amodal_validation.json       # 3,600 ảnh, 15,654 nhãn (validation/test)
```
> **Tổng số mẫu train:** $690 + 12,376 = 13,066$ instances.  
> **Lưu ý về tập Test:** Ban tổ chức MVTec không công khai nhãn ground-truth của tập Test (`D2S_amodal_test_info.json` không có annotation). Theo chuẩn chung của cộng đồng nghiên cứu trên D2SA, tập **Validation** được dùng làm tập kiểm thử để đánh giá các chỉ số mIoU.

#### 4. Lệnh Kiểm tra Nhanh Tính Toàn Vẹn Dataset (Trước khi chạy 11+ giờ):
Chỉ cần chạy lệnh kiểm tra tự động sau để đảm bảo không bị thiếu ảnh hoặc file JSON hỏng:
```bash
python scripts/verify_d2sa_dataset.py
```
*Script sẽ kiểm tra sự tồn tại của cả 3 file JSON, đếm đủ 22,562 ảnh, và đọc thử ngẫu nhiên một số ảnh. Nếu in ra `✅ KẾT QUẢ: TẬP DỮ LIỆU HOÀN TOÀN HỢP LỆ VÀ SẴN SÀNG HUẤN LUYỆN!`, thầy có thể yên tâm cho chạy full training.*

---

### 3.2. Chạy Cấu hình Chính (Row 4 - Main Config)

**Huấn luyện 30 Epochs trên GPU A100:**
```bash
python scripts/train_d2sa.py \
    --epochs 30 \
    --batch-size 4 \
    --accumulation-steps 4 \
    --num-workers 4 \
    --lr 1e-4 \
    --device cuda
```

**Đánh giá checkpoint:**
```bash
python scripts/evaluate_d2sa.py \
    --checkpoint checkpoints/d2sa/amodal_shape_prediction_main_config/swin_amodal_epoch_30.pth \
    --device cuda \
    --output results/d2sa/row4_eval.json
```

---

### 3.3. Chạy Toàn bộ 6 Cấu hình Bằng Runner Tự Động (Khuyến nghị cho A100)

**Cách 1: Chạy trực tiếp qua Bash Script (chạy trong `tmux` hoặc nền):**
```bash
# Cú pháp: bash scripts/run_ablation_d2sa.sh <device> <epochs> <batch_size> <acc_steps> <num_workers>
bash scripts/run_ablation_d2sa.sh cuda 30 4 4 4
```

**Cách 2: Chạy qua Python Runner (hỗ trợ chọn lọc cấu hình):**
```bash
# Chạy toàn bộ 6 cấu hình:
python scripts/run_ablation_d2sa.py --rows 1 2 3 4 5 6 --mode all --device cuda --epochs 30 --num-workers 4

# Hoặc chỉ chạy riêng các cấu hình có sự khác biệt (ví dụ Row 4, 5, 6):
python scripts/run_ablation_d2sa.py --rows 4 5 6 --mode all --device cuda --epochs 30
```

**Cách 3: Chạy nền qua `nohup` (để ngắt kết nối SSH an toàn):**
```bash
nohup bash scripts/run_ablation_d2sa.sh cuda 30 4 4 4 > ablation_runner.log 2>&1 &
echo $! > ablation_runner.pid
```

---

### 3.4. Chạy Từng Cấu hình Ablation Riêng lẻ

#### Row 1 (4 kênh, BCE Loss, Không Edge, Không Emb, Không Spatial):
```bash
python scripts/other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/train.py --device cuda --epochs 30
python scripts/other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/evaluate.py --device cuda
```

#### Row 2 (5 kênh, BCE Loss, Có Edge, Không Emb, Không Spatial):
```bash
python scripts/other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/train.py --device cuda --epochs 30
python scripts/other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/evaluate.py --device cuda
```

#### Row 3 (5 kênh, Occ-Aware Loss 5x, Có Edge, Không Emb, Không Spatial):
```bash
python scripts/other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding/train.py --device cuda --epochs 30
python scripts/other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding/evaluate.py --device cuda
```

#### Row 5 (5 kênh, Occ-Aware Loss 5x, Có Edge, Không Emb, Có Spatial Attention):
```bash
python scripts/other_config_d2sa/amodal_shape_prediction_no_embeding/train.py --device cuda --epochs 30
python scripts/other_config_d2sa/amodal_shape_prediction_no_embeding/evaluate.py --device cuda
```

#### Row 6 (Full Config: 5 kênh, Occ-Aware Loss 5x, Có Edge, Có Emb 60, Có Spatial Attention):
```bash
python scripts/other_config_d2sa/amodal_shape_prediction_full_config/train.py --device cuda --epochs 30
python scripts/other_config_d2sa/amodal_shape_prediction_full_config/evaluate.py --device cuda
```

---

## ⏱️ 4. Ước tính Thời gian & Ngoại suy Hiệu năng Huấn luyện

> ⚠️ **LƯU Ý QUAN TRỌNG:** Các con số thời gian dưới đây là **ƯỚC LƯỢNG NGOẠI SUY** dựa trên bài đo thực nghiệm 200 mẫu trên GPU máy trạm cục bộ, **chưa phải số đo trực tiếp trên phần cứng A100**. Do ảnh gốc D2SA có độ phân giải lớn (1440×1920) kèm khâu giải mã RLE mask và tăng cường hình ảnh (Augmentation), khâu nạp dữ liệu (I/O & CPU) chính là nút thắt cổ chai lớn nhất, không phải tốc độ tính toán của GPU.

### 4.1. Bằng chứng Đo lường Thực nghiệm Cục bộ (Subset 200 mẫu):
- **Phần cứng thử nghiệm:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM), CPU Intel Core i5/i7, ổ cứng SSD thông thường.
- **Tham số đo:** Batch Size = 4, Accumulation = 1, Num Workers = 2, Resize = 224×224.
- **Kết quả đo đạc thực tế:**
  - Thời gian xử lý 200 mẫu (50 batch): **68.95 giây**.
  - Tốc độ xử lý thực tế: **~2.90 mẫu / giây** (~0.725 batch / giây).
  - Phân tích log: Có hiện tượng ngắt quãng 1–5 giây giữa các lượt worker đọc đĩa giải mã ảnh 1440×1920 và RLE polygon, sau đó GPU forward/backward rất nhanh.

### 4.2. Bảng Ngoại suy Thời gian Huấn luyện Tập D2SA Đầy đủ (13,066 mẫu / epoch):

| Môi trường & Thiết lập | Throughput ước tính | Thời gian / Epoch | 1 Cấu hình (30 Epochs) | Toàn bộ 6 Cấu hình (Train thuần) | Ghi chú & Trạng thái |
|:---|:---:|:---:|:---:|:---:|:---|
| **Local RTX 3050** (2 Workers, Batch 4) | ~2.90 mẫu/s | ~75 phút | ~37.5 giờ | ~225 giờ | *Đã đo thực nghiệm (Chỉ nên chạy subset)* |
| **A100 Server** (4 Workers, NVMe SSD) | ~15 – 25 mẫu/s | ~8.5 – 14.5 phút | ~4.2 – 7.2 giờ | ~25 – 43 giờ | *Ngoại suy (Worker vừa phải)* |
| **A100 Server** (8–16 Workers, NVMe SSD) | ~35 – 50 mẫu/s | ~4.3 – 6.2 phút | ~2.1 – 3.1 giờ | **~13 – 18 giờ** | *Ngoại suy khuyến nghị (Bão hòa GPU)* |

> 💡 **Khuyến nghị đặc biệt dành cho Thầy khi chạy trên A100:**
> 1. Khi bắt đầu chạy, thầy hãy gõ lệnh `watch -n 1 nvidia-smi` để theo dõi mức độ tải của GPU (**GPU-Util**).
> 2. Nếu thấy **GPU-Util dưới 50%**, điều đó có nghĩa là GPU đang phải chờ CPU giải mã ảnh 1440×1920. Thầy hãy tăng số luồng nạp dữ liệu lên `--num-workers 8` hoặc `--num-workers 16` để tận dụng nhiều nhân CPU trên server A100, giúp GPU đạt công suất tối đa.
> 3. VRAM tiêu thụ ở cấu hình Batch 4 chỉ khoảng **3.8 GB**, hoàn toàn an toàn trên A100 (40GB/80GB).

---

### 4.3. Dự toán Thời gian Đánh giá Validation (Evaluation Overhead) & Chiến lược Đánh giá 2 Tầng

> ⚠️ **ĐIỂM CỐT LÕI:** Dự toán 13–18 giờ ở trên **chưa bao gồm thời gian evaluate**. Nếu không có chiến lược hợp lý, thời gian đánh giá sẽ làm đội thêm **3–5 giờ GPU**!

#### 1. Bằng chứng Đo lường Thực nghiệm Tốc độ Đánh giá (Đo trực tiếp trên tập Validation D2SA):
- **Phần cứng thử nghiệm:** NVIDIA GeForce RTX 3050 Laptop GPU, `DataLoader(batch_size=4, num_workers=0)`.
- **Kết quả đo:**
  - Thời gian evaluate 100 mẫu validation: **7.40 giây**.
  - Throughput đánh giá thực tế: **~13.51 mẫu / giây**.
  - Ngoại suy trên máy local cho full 15,654 mẫu validation: $\frac{15,654}{13.51} \approx 1,158.8\text{s} \approx \mathbf{19.3\text{ phút / lượt}}$.
- **Ngoại suy trên máy chủ A100 (với NVMe SSD và DataLoader 8–16 workers):**
  - Throughput ước tính: **~40 – 65 mẫu / giây**.
  - Thời gian chạy 1 lượt đánh giá toàn bộ 15,654 mẫu: $\frac{15,654}{40..65} \approx \mathbf{4.0 – 6.5\text{ phút / lượt}}$.

#### 2. Phân tích Rủi ro Quá tải Thời gian nếu Đánh giá Toàn bộ Định kỳ:
- Mặc định `--eval-every 5` cộng thêm lượt đánh giá cuối (tại các Epoch 5, 10, 15, 20, 25, 30 $\to$ **6 lượt đánh giá full / cấu hình**):
  - Mỗi cấu hình tốn thêm: $6 \times (4.0 – 6.5\text{ phút}) \approx \mathbf{24 – 39\text{ phút}}$.
  - Cả 6 cấu hình tốn thêm: $6 \times (24 – 39\text{ phút}) \approx \mathbf{2.4 – 3.9\text{ giờ}}$.
  - Tổng thời gian huấn luyện + đánh giá sẽ bị đội lên **16 – 22 giờ**.

#### 3. Giải pháp Tối ưu: Đánh giá 2 Tầng (`--val-subset-size 500`):
Hệ thống cả 6 cấu hình đã được nâng cấp cơ chế **Dual-level Validation DataLoader**:
- **Các lượt định kỳ giữa chừng (Epoch 5, 10, 15, 20, 25):** Chạy trên tập con `--val-subset-size 500` mẫu.
  - Thời gian mỗi lượt trên A100: **~8 – 12 giây** (gần như tức thì!).
  - Mục đích: Theo dõi tiến trình hội tụ, kiểm tra chỉ số `% pixel dương` (`pos_pixel_pct`) để bắt sớm lỗi sụp đổ mô hình mà không làm gián đoạn GPU.
- **Lượt đánh giá tổng kết ở Epoch 30 (Epoch cuối cùng):** Tự động chuyển sang nạp toàn bộ **15,654 mẫu validation** để tính toán bộ chỉ số mIoU, Dice, Precision, Recall chính xác tuyệt đối.
  - Thời gian đánh giá epoch cuối: **~5 – 6 phút / cấu hình**.
  - **Tổng thời gian tiết kiệm:** Cắt giảm từ ~3.5 giờ xuống chỉ còn **~40–45 phút** cho cả 6 cấu hình!

#### 4. Quy chuẩn Báo cáo Khoa học (Scientific Protocol):
> 🎯 **NGUYÊN TẮC BÁO CÁO KHOA HỌC:**
> - Ban tổ chức MVTec không cung cấp nhãn ground-truth cho tập Test D2SA (`D2S_amodal_test_info.json` không có annotation), nên theo thông lệ quốc tế, tập Validation D2SA (15,654 mẫu) chính là tập benchmark báo cáo số liệu.
> - Trong báo cáo nghiệm thu và bài báo, **BẮT BUỘC dùng kết quả tại Epoch 30** (giống hệt quy chuẩn đã thực hiện trên COCOA và KINS).
> - **TUYỆT ĐỐI KHÔNG chọn "Epoch tốt nhất trên validation" (Best Validation Epoch):** Vì tập validation chính là tập kiểm thử cuối cùng, việc cherry-pick epoch có điểm cao nhất sẽ tạo ra hiện tượng rò rỉ dữ liệu (data leakage) và thiên kiến lạc quan (optimistic bias), làm mất tính công bằng khi so sánh với COCOA và KINS.

---

## 🔄 5. Cơ chế Resume Liền mạch & Chiến lược Checkpoint Tiết kiệm Quota

Khi chạy trên cụm máy chủ chia sẻ hoặc hệ thống quản lý hàng đợi (Slurm, Kubernetes, Colab Timeout), tiến trình có thể bị gửi tín hiệu `SIGTERM` để thu hồi tài nguyên. Hệ thống áp dụng chiến lược checkpoint **2 tầng tối ưu dung lượng** kết hợp phục hồi trạng thái toàn diện:

### 5.1. Chiến lược Checkpoint 2 Tầng & Ghi Nguyên Tử (Atomic Write):
Nếu lưu toàn bộ trọng số + optimizer cho cả 30 epoch $\times$ 6 cấu hình, dung lượng đĩa sẽ lên tới **~73 GB**, dễ gây tràn quota tài khoản trên server trường và làm chết tiến trình. Do đó hệ thống phân tách:
1. **Checkpoint định kỳ mỗi epoch (`swin_amodal_epoch_X.pth`):**
   - Chỉ lưu `model.state_dict()` (~131 MB/file).
   - Mục đích: Dùng để chạy `evaluate_d2sa.py` đánh giá mIoU của từng epoch sau khi huấn luyện xong.
2. **Checkpoint duy trì tiến trình (`last.pth` - Ghi nguyên tử):**
   - Lưu đầy đủ `{'epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'loss'}` (~405 MB).
   - **Ghi nguyên tử (Atomic write):** Để tránh nguy cơ hỏng checkpoint nếu tiến trình bị `SIGKILL` đúng lúc đang ghi file 405 MB (khiến cả bản mới bị hỏng dở và bản cũ của epoch trước bị mất), hệ thống luôn ghi ra `last.pth.tmp`, thực hiện `flush()` và `os.fsync()`, sau đó dùng `os.replace("last.pth.tmp", "last.pth")`. Thao tác thay thế này là nguyên tử ở tầng hệ điều hành, đảm bảo `last.pth` luôn là bản toàn vẹn của epoch trước hoặc bản toàn vẹn của epoch mới.
   - **Tự động Fallback khi nạp:** Nếu `torch.load` gặp bất kỳ lỗi nào khi đọc `last.pth`, chương trình sẽ ghi cảnh báo vào log và tự động rơi về `swin_amodal_epoch_X.pth` kết hợp tua nhanh (`fast-forward`) scheduler để tiếp tục huấn luyện mà không bị crash.
3. **Checkpoint cứu hộ khi nhận SIGTERM (`emergency_checkpoint_epoch_X.pth`):**
   - Tự động bắt tín hiệu `SIGTERM` (từ Queue Manager Slurm/PBS khi hết grace period) và lưu đầy đủ cả model + optimizer + scheduler qua cơ chế nguyên tử kèm `os.fsync()` trong < 0.5s.

> 💾 **Tổng kết dung lượng ổ đĩa:**
> - Mỗi cấu hình: $30 \times 131\text{ MB} + 405\text{ MB} \approx 4.3\text{ GB}$.
> - Cả 6 cấu hình: $4.3\text{ GB} \times 6 \approx \mathbf{26\text{ GB}}$ (tiết kiệm **~47 GB** so với cách lưu thông thường).
> - ⚠️ **Khuyến nghị cho Thầy:** Thầy nên gõ lệnh `quota -s` hoặc `df -h` để kiểm tra dung lượng còn trống của tài khoản trên máy chủ, đảm bảo có tối thiểu **~30 GB** trước khi bắt đầu.

### 5.2. Trạng thái Khôi phục khi Resume:
- **Thành phần được bảo toàn 100%:** Trọng số mô hình (`model`), hai vector moment $m_t$ và $v_t$ của bộ tối ưu (`optimizer` AdamW), và bước giảm tốc độ học (`scheduler` CosineAnnealingLR).
  - *Lợi ích:* Triệt tiêu hiện tượng vọt loss (loss spike) ở các step đầu tiên sau khi resume.
- **Thành phần không khôi phục:** Trạng thái bộ sinh số ngẫu nhiên (RNG state của DataLoader shuffle và các phép biến đổi ảnh ngẫu nhiên như ShiftScaleRotate, RandomBrightness). Do đó, run bị resume sẽ có thứ tự batch và augmentation khác đôi chút so với run chạy liền mạch, nhưng sự khác biệt ngẫu nhiên này không làm suy giảm chất lượng hội tụ hay tính khách quan của thực nghiệm.

### 5.3. Quy Tắc Duy Nhất và Nhất Quán của `--resume-epoch`:
Để tránh nhầm lẫn và sai lệch lộ trình learning rate, hệ thống áp dụng **một quy tắc duy nhất**:
> 🎯 **NGUYÊN TẮC:** 
> - Checkpoint luôn ghi nhận `last_completed_epoch` (Epoch đã hoàn tất 100% gần nhất).
> - Cờ `--resume-epoch <N>` nhận vào `N = last_completed_epoch`.
> - Khi khởi động lại, tiến trình luôn bắt đầu từ **Epoch $N + 1$** và **chạy lại toàn bộ epoch bị gián đoạn**.
> - Scheduler được đồng bộ đúng $N$ bước, đảm bảo sau khi chạy xong epoch bị gián đoạn, learning rate nằm chính xác trên đường Cosine.

**Ví dụ cụ thể cho Thầy khi thao tác:**
- Giả sử tiến trình đang chạy dở ở **Epoch 14** thì bị ngắt (bị kill hoặc hết thời gian GPU):
  - Lúc này, Epoch đã hoàn tất trọn vẹn gần nhất là **Epoch 13** (`last_completed_epoch = 13`).
  - Hệ thống tự động lưu: `emergency_checkpoint_epoch_13.pth` (và `last.pth` chứa metadata `epoch: 13`).
  - Log hướng dẫn hiển thị rõ ràng:
    ```text
    💡 [RESUME GUIDE] Để chạy lại toàn bộ Epoch 14 bị gián đoạn, hãy chạy: --resume-epoch 13
    ```
  - **Lệnh Thầy cần gõ để tiếp tục:**
    ```bash
    python scripts/train_d2sa.py --resume-epoch 13 --epochs 30 --device cuda
    ```
  - **Diễn biến bên dưới hệ thống:**
    1. Tiến trình nạp checkpoint `emergency_checkpoint_epoch_13.pth` (hoặc `last.pth` khớp epoch 13).
    2. Khôi phục trọng số, optimizer và scheduler tại mốc cuối Epoch 13.
    3. Chạy lại toàn bộ **Epoch 14**, sau đó tiếp tục Epoch 15 $\to$ 30.
    4. Learning rate ở Epoch 14, 15, ..., 30 khớp 100% không lệch một nấc nào so với một run chạy liên tục không ngắt.

### 5.4. Xác Nhận Siêu Tham Số Huấn Luyện & Đối Chiếu Learning Rate Resume Tuyệt Đối:
- **Cam kết "Giữ y hệt hyperparameter" của cả 3 dataset (COCOA, KINS, D2SA):**
  - Toàn bộ 6 script D2SA và 6 script COCOA/KINS đều sử dụng:
    - Optimizer: `AdamW(lr=1e-4, weight_decay=1e-4)`
    - Scheduler: `CosineAnnealingLR(optimizer, T_max=30)` với giá trị `eta_min = 0` (giá trị mặc định của PyTorch, không can thiệp).
- **Đối chiếu công thức toán học và kiểm chứng thực nghiệm trên script thật:**
  Công thức cập nhật của PyTorch `CosineAnnealingLR` với `eta_min = 0`:
  $$\text{LR}_N = \eta_{\min} + \frac{1}{2}(\text{LR}_{\text{base}} - \eta_{\min}) \left(1 + \cos\left(\frac{N \pi}{T_{\max}}\right)\right) = 10^{-4} \times \frac{1 + \cos\left(\frac{N \pi}{30}\right)}{2}$$
  Đối chiếu số liệu sau khi hoàn tất $N$ epoch giữa giá trị kỳ vọng của User, công thức toán, và lệnh chạy thực tế trên `scripts/train_d2sa.py --resume-epoch N`:

| $N$ (Epoch đã hoàn tất) | LR Kỳ vọng (User) | LR PyTorch / Script thật | Độ lệch (Delta) | Trạng thái đối chiếu |
|:---:|:---:|:---:|:---:|:---:|
| **1** | `9.972609e-05` | `9.97260948e-05` | $4.77 \times 10^{-12}$ | ✅ Khớp chính xác tuyệt đối |
| **13** | `6.039558e-05` | `6.03955845e-05` | $4.54 \times 10^{-12}$ | ✅ Khớp chính xác tuyệt đối |
| **14** | `5.522642e-05` | `5.52264232e-05` | $3.16 \times 10^{-12}$ | ✅ Khớp chính xác tuyệt đối |
| **30** | `0` | `0.000000e+00` | $0.00$ | ✅ Khớp chính xác tuyệt đối |

*Log chạy thực tế từ `scripts/train_d2sa.py` khi tua nhanh (fast-forward) scheduler đã in ra đúng từng giá trị trên, chứng minh script thật nạp đúng 100% cấu hình.*

---

## 📈 6. Chỉ số "% Pixel Dương" (`PosPixels` / `pos_pixel_pct`)

Nhằm phát hiện sớm hiện tượng **sụp đổ mô hình (model collapse)** ngay từ những epoch đầu tiên (thay vì phải đợi sau 30 epoch mới phát hiện), hệ thống ghi nhận chỉ số tỷ lệ phần trăm pixel dự đoán là tiền cảnh (`sigmoid(output) > 0.5`):

```text
[EPOCH 1/30] Loss: 1.5832 | Time: 182.4s | LR: 9.97e-05 | PosPixels: 14.85% | Val: [mIoU: 0.1820, ..., pos_pixel_pct: 12.40%]
```

- **Ý nghĩa:**
  - **`PosPixels` (Train):** Trung bình tỷ lệ % pixel dương trên toàn bộ các batch trong epoch huấn luyện.
  - **`pos_pixel_pct` (Val):** Tỷ lệ % pixel dương trên toàn bộ tập validation.
- **Quy tắc chẩn đoán nhanh:**
  - `PosPixels ≈ 8% - 20%`: Mô hình dự đoán tự nhiên, phân bố khớp với ground-truth (D2SA có tỷ lệ foreground amodal trung bình khoảng 6.6%).
  - `PosPixels = 0.00%`: Mô hình sụp đổ về lớp nền (Background Collapse - dự đoán toàn bộ pixel là 0).
  - `PosPixels = 100.00%`: Mô hình sụp đổ về toàn bộ 1 (Foreground Collapse).

---

## 🛡️ 7. Giám sát & Bền vững Dữ liệu (Telemetry)

Toàn bộ 6 cấu hình lưu trữ song song tại `logs/d2sa/`:
```text
logs/d2sa/
├── row1_bce_vis_noedge_noemb_nospatial/
├── row2_bce_vis_edge_noemb_nospatial/
├── row3_occ_vis_edge_noemb_nospatial/
├── row4_main_config/
├── row5_occ_vis_edge_noemb_spatial/
└── row6_full_config/
```

Mỗi lượt chạy tạo 2 file song song:
1. `*_train.log`: Log text định dạng chuẩn, ghi nhận môi trường GPU/CUDA, thông số tham số, git hash, thời gian từng epoch, traceback exception đầy đủ.
2. `*_metrics.jsonl`: File JSON Lines (mỗi dòng 1 JSON object) cho phép theo dõi thời gian thực bằng code hoặc import vào Pandas / Weights & Biases.

**Lệnh theo dõi tiến trình trực tiếp:**
```bash
tail -f logs/d2sa/row4_main_config/*_train.log
```

---

## 🔬 8. Báo Cáo Kiểm Tra Toàn Vẹn Dữ Liệu D2SA (Phase 3 Data Integrity)

Trước khi tiến hành nhân bản 6 cấu hình, 2 phép kiểm tra độc lập chuyên sâu đã được thực hiện để loại trừ hoàn toàn các lỗi âm thầm (silent bugs):

### 8.1. Kiểm Tra Xung Đột `image_id` Khi Gộp Dữ Liệu (`AmodalDatasetD2SA_Concat`):
- **Bối cảnh:** Tập train D2SA được gộp từ 2 file JSON độc lập: `D2S_amodal_training_rot0.json` và `D2S_amodal_augmented.json`.
- **Kết quả kiểm tra:**
  - `D2S_amodal_training_rot0.json`: 438 ảnh, dải `image_id` từ **200** đến **44,520**.
  - `D2S_amodal_augmented.json`: 1,562 ảnh, dải `image_id` từ **88,000,000** đến **88,001,561**.
  - Số lượng `image_id` trùng nhau giữa 2 tập: **0 ảnh** (Hoàn toàn tách biệt không gian ID).
- **Kết luận:** An toàn tuyệt đối 100%, không xảy ra hiện tượng gán nhầm annotation giữa các ảnh khi huấn luyện chung.

### 8.2. Kiểm Tra Trực Quan Hình Học & Khớp Mặt Nạ (4 Mẫu Tỷ Lệ Che 20–60%, Ảnh Cần Pad):
- **Bối cảnh:** D2SA có **1,556 kích thước ảnh khác nhau** do zero-padding khi vật thể tràn biên. Cần xác nhận không bị lỗi transpose $H/W$, flip hoặc lệch offset giữa ảnh RGB và các loại mặt nạ.
- **Quy chuẩn hiển thị:** Đầy đủ 4 ô: (1) RGB, (2) Visible overlay (xanh lá), (3) Amodal overlay (đỏ) kèm vùng bị che (xanh dương), (4) Edge mask.
- **Mẫu kiểm tra:** Đã trích xuất 4 mẫu đại diện thuộc dải che khuất điển hình 20%–60%, trong đó có 2 mẫu kích thước khác 1440×1920 (ảnh cần zero-padding):
  1. **Mẫu 1 (`ethiquable_gruener_tee_ceylon`):**
     - Ảnh `D2S_000720.jpg` | Kích thước gốc: **1548 × 1920** (Kích thước khác 1440×1920, cần zero-padding).
     - Tỷ lệ che: **38.3%** (Visible: 24,015 px, Amodal: 38,913 px).
     - Hình ảnh 4 ô: [`d2sa_visual_check_sample_1.png`](../assets/d2sa_visual_checks/d2sa_visual_check_sample_1.png)
  2. **Mẫu 2 (`kilimanjaro_tea_earl_grey`):**
     - Ảnh `D2S_000721.jpg` | Kích thước gốc: **1457 × 1920** (Kích thước khác 1440×1920, cần zero-padding).
     - Tỷ lệ che: **43.3%** (Visible: 22,231 px, Amodal: 39,208 px).
     - Hình ảnh 4 ô: [`d2sa_visual_check_sample_2.png`](../assets/d2sa_visual_checks/d2sa_visual_check_sample_2.png)
  3. **Mẫu 3 (`cocoba_fruehstueckskakao_mit_honig`):**
     - Ảnh `D2S_000000.jpg` | Kích thước: **1440 × 1920** (Kích thước chuẩn).
     - Tỷ lệ che: **36.6%** (Visible: 40,862 px, Amodal: 64,482 px).
     - Hình ảnh 4 ô: [`d2sa_visual_check_sample_3.png`](../assets/d2sa_visual_checks/d2sa_visual_check_sample_3.png)
  4. **Mẫu 4 (`gepa_bio_und_fair_kamillentee`):**
     - Ảnh `D2S_000000.jpg` | Kích thước: **1440 × 1920** (Kích thước chuẩn).
     - Tỷ lệ che: **32.5%** (Visible: 26,450 px, Amodal: 39,183 px).
     - Hình ảnh 4 ô: [`d2sa_visual_check_sample_4.png`](../assets/d2sa_visual_checks/d2sa_visual_check_sample_4.png)
- **Kết quả xác nhận:**
  - `visible_mask` (xanh lá): Khớp khít từng pixel với phần lộ ra của vật thể trên RGB.
  - `amodal_mask` (đỏ): Bao trùm hoàn chỉnh toàn bộ biên dạng thật của vật thể (kể cả phần bị che).
  - `occluded_mask` (xanh dương): Phản ánh chính xác phần bị che khuất ($Amodal \setminus Visible$).
  - `edge_mask`: Bám sát theo đúng đường biên của visible mask.
  - Tỷ lệ `iscrowd`: **0/28,720** (toàn bộ là vật thể đơn lẻ).
  - Ánh xạ `category_id`: 60 lớp (1..60) được chuyển đổi chính xác sang index `0..59` cho `nn.Embedding`.
