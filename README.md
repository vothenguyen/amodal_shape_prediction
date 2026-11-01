# 👁️ Amodal Shape Prediction (Swin-UNet + SAM 2.1)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue.svg?style=flat-square&logo=pytorch)
![Datasets](https://img.shields.io/badge/Datasets-COCOA%20%7C%20KINS%20%7C%20D2SA-green.svg?style=flat-square)
![GPU Ready](https://img.shields.io/badge/GPU-RTX%203050%20%7C%20A100-orange.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)

Dự án Nghiên cứu Khoa học / Đồ án Chuyên ngành Khoa học Máy tính: **Dự đoán biên dạng hoàn chỉnh của vật thể bị che khuất (Amodal Instance Segmentation & Shape Prediction)**.

Hệ thống được xây dựng theo kiến trúc Pipeline 2 giai đoạn (2-Stage) kết hợp sức mạnh Zero-shot của **Segment Anything Model (SAM 2.1)** với mạng nơ-ron tích hợp biểu diễn toàn cục **Amodal Swin-UNet**, hỗ trợ đầy đủ 3 tập dữ liệu chuẩn benchmark quốc tế: **COCOA**, **KINS**, và **D2SA (MVTec D2S Amodal)**.

---

## 🌟 1. Tính năng & Điểm nhấn Kỹ thuật

- **Pipeline 2-Stage mạnh mẽ:**
  - *Stage 1 (Phân đoạn cục bộ):* `SAM 2.1` trích xuất mặt nạ nhìn thấy (*Visible Mask*) từ tương tác chuột (Point Prompt) hoặc bộ phát hiện (Detector).
  - *Stage 2 (Suy luận Amodal):* `Swin-UNet` 5 kênh phân tích đa nguồn thông tin để tái tạo hình dáng trọn vẹn (*Amodal Mask*) và vùng bị che khuất (*Occluded Region*).
- **Hỗ trợ 3 Tập Benchmark Lớn:**
  - **COCOA (COCO Amodal):** 91 lớp đối tượng tự nhiên với trường thứ tự xếp chồng (`order`).
  - **KINS (KITTI In-house):** 8 lớp phương tiện giao thông đường phố.
  - **D2SA (MVTec D2S Amodal):** 60 lớp sản phẩm công nghiệp và bán lẻ, 22,562 ảnh độ phân giải cao ($1440 \times 1920$).
- **Hệ thống Logging & Telemetry Bền vững:**
  - Ghi song song log console, file `*_train.log`, và `*_metrics.jsonl` (tự động flush & `os.fsync`).
  - Theo dõi sớm lỗi sụp đổ mô hình (*Model Collapse*) qua chỉ số `% Pixel Dương` (`PosPixels`).
- **Khả năng Phục hồi Lỗi Toàn diện (Fault-Tolerant Training trên A100/Slurm):**
  - **Ghi nguyên tử (Atomic Checkpoint):** `last.pth.tmp` $\to$ `fsync` $\to$ `os.replace` loại bỏ hoàn toàn nguy cơ hỏng checkpoint khi hết quota hoặc bị kill.
  - **Cứu hộ tín hiệu SIGTERM:** Tự động bắt tín hiệu kết thúc từ Queue Manager (Slurm/PBS) và lưu toàn bộ trạng thái (Model + AdamW moments + Scheduler) trong $< 0.5\text{s}$.
  - **Resume liền mạch (`--resume-epoch`):** Thuật toán tua nhanh (*fast-forward*) scheduler đưa tốc độ học khớp chính xác $100\%$ theo đường cong `CosineAnnealingLR`.
- **Chiến lược Đánh giá 2 Tầng (Dual-level Validation):**
  - Đánh giá định kỳ giữa chừng trên subset `--val-subset-size 500` (~10s/lượt) để theo dõi độ hội tụ.
  - Đánh giá toàn bộ 15,654 mẫu validation D2SA ở Epoch 30 (~5–6 phút) phục vụ báo cáo khoa học, tiết kiệm gần 3 giờ GPU.

---

## 🧠 2. Kiến trúc Mô hình (Amodal Swin-UNet)

Mô hình nâng cấp cấu trúc đầu vào đa kênh nhằm gia tăng tối đa *Inductive Bias*:

```text
[Ảnh RGB (3ch)] ──┐
[Visible Mask (1ch)] ─┼─► Tensor [B, 5, 224, 224] ─► Swin-UNet Encoder (Patch Embed 5ch) 
[Edge Mask (1ch)] ────┘                                      │
                                                             ▼ (Bottleneck 768ch)
[Category ID] ─────────► nn.Embedding(N, 768) ───────────► (+) Cộng vector ngữ nghĩa
                                                             │
                                                             ▼
                                                    U-Net Decoder (3 UpBlocks)
                                                             │
                                                             ▼ (Tùy chọn Spatial Attention)
                                                    Head Conv2d(64, 1, 1) ─► Amodal Mask Logits
```

- **Encoder:** Backbone `swin_tiny_patch4_window7_224` (timm pretrained), sửa đổi Patch Embedding nhận 4 hoặc 5 kênh.
- **Edge Mask:** Ranh giới phần nhìn thấy của vật thể, trích xuất bằng toán tử hình thái học (`cv2.dilate` / `erode`).
- **Category Embedding:** Chuyển đổi mã định danh lớp (`category_id`) thành vector 768 chiều cộng vào bottleneck.
- **Hàm Loss Occlusion-Aware:** Phạt $5\times$ trọng số trên vùng bị che khuất ($Amodal \setminus Visible$) kết hợp Soft Dice Loss.

---

## 📊 3. Nghiên cứu Thực nghiệm Ablation Study (Table V)

Dự án hỗ trợ đầy đủ 6 cấu hình ablation study đối chiếu chuẩn xác giữa mã nguồn và lý thuyết toán học:

| Row | Tên cấu hình | Kênh vào | Edge Mask | Category Embedding | Spatial Attention | Hàm Loss | Số tham số |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce` | **4** (RGB+Vis) | ❌ | ❌ | ❌ | BCEWithLogits | **34,352,027** |
| **2** | `amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce` | **5** (RGB+Vis+Edge) | ✅ | ❌ | ❌ | BCEWithLogits | **34,353,563** |
| **3** | `amodal-shape-prediction-no-spatial-no-embeding` | **5** (RGB+Vis+Edge) | ✅ | ❌ | ❌ | Occ-Aware (5x) + Dice | **34,353,563** |
| **4** | **Cấu hình chính (Main / Baseline)** | **5** (RGB+Vis+Edge) | ✅ | ✅ (60 lớp) | ❌ | Occ-Aware (5x) + Dice | **34,399,643** |
| **5** | `amodal_shape_prediction_no_embeding` | **5** (RGB+Vis+Edge) | ✅ | ❌ | ✅ ($k=7$) | Occ-Aware (5x) + Dice | **34,353,661** |
| **6** | `amodal_shape_prediction_full_config` | **5** (RGB+Vis+Edge) | ✅ | ✅ (60 lớp) | ✅ ($k=7$) | Occ-Aware (5x) + Dice | **34,399,741** |

> 📐 **Đối chiếu Toán học & Assertion:**
> - $\Delta_{\text{Row2-Row1}} = +1,536$ tham số (Thêm 1 kênh vào Patch Embedding: $96 \times 1 \times 4 \times 4$).
> - $\Delta_{\text{Row4-Row3}} = +46,080$ tham số (Category Embedding cho 60 lớp: $60 \times 768$).
> - $\Delta_{\text{Row5-Row3}} = +98$ tham số (Spatial Attention Conv2d $7 \times 7$: $1 \times 2 \times 7 \times 7$).
> - Cả 6 script đều tích hợp lệnh `assert total_params == EXPECTED_PARAMS` khi khởi động để loại trừ sai sót cấu hình.

---

## 🗂️ 4. Cấu trúc Mã nguồn Dự án

```text
.
├── app.py                     # Giao diện tương tác trực quan (Gradio Web UI)
├── requirements.txt           # Thư viện phụ thuộc
├── .gitattributes             # Chuẩn hóa ngắt dòng LF cho file .sh
├── a100_training_handoff_guide.md # Tài liệu nghiệm thu Phase 6 & hướng dẫn máy chủ A100
├── assets/
│   ├── Table.png              # Bảng Table V Ablation Study gốc
│   ├── figures/               # Sơ đồ kiến trúc mô hình
│   └── d2sa_visual_checks/    # Ảnh kiểm tra hình học và khớp mặt nạ (4 panels)
├── checkpoints/               # Checkpoint mô hình (.pth)
│   └── d2sa/                  # Checkpoint cho 6 cấu hình D2SA
├── data/                      # Thư mục dữ liệu (COCOA, KINS, D2SA)
├── docs/                      # Tài liệu báo cáo, giải thích chiến lược occlusion
├── logs/                      # Telemetry logs (*_train.log, *_metrics.jsonl)
├── results/                   # Kết quả đánh giá định lượng JSON
└── scripts/                   # Mã nguồn huấn luyện & đánh giá
    ├── model.py               # Kiến trúc Swin-UNet chuẩn cho COCOA (Row 4)
    ├── dataset.py             # Dataset loader cho COCOA (đọc order, amodal polygon)
    ├── train.py               # Huấn luyện Row 4 trên COCOA
    ├── evaluate.py            # Đánh giá COCOA
    ├── dataset_d2sa.py        # Dataset loader D2SA (giải mã RLE, gộp training+augmented)
    ├── train_d2sa.py          # Huấn luyện Row 4 Main Config trên D2SA
    ├── evaluate_d2sa.py       # Đánh giá D2SA (mIoU, Dice, Precision, Recall, Inv-mIoU)
    ├── verify_d2sa_dataset.py # Script kiểm tra nhanh tính toàn vẹn dataset D2SA
    ├── logging_utils.py       # Module logging, SIGTERM handling, atomic save
    ├── run_ablation_d2sa.py   # Runner Python tự động chạy 6 cấu hình D2SA
    ├── run_ablation_d2sa.sh   # Runner Bash cho server Linux / A100 / Slurm
    ├── README.md              # Hướng dẫn kỹ thuật chuyên sâu Ablation Study
    ├── other_config_cocoa/    # 5 cấu hình ablation trên COCOA
    └── other_config_d2sa/     # 5 cấu hình ablation trên D2SA
```

---

## 🚀 5. Hướng dẫn Chạy Huấn luyện & Đánh giá

### 5.1. Cài đặt Môi trường
```bash
pip install torch torchvision timm albumentations opencv-python tqdm pycocotools numpy matplotlib gradio
```

### 5.2. Chuẩn bị Dữ liệu D2SA
Xem hướng dẫn chi tiết tải trực tiếp dữ liệu từ trang MVTec trong [scripts/README.md](scripts/README.md#31-chuẩn-bị-môi-trường--tải-dữ-liệu-d2sa-bắt-buộc).  
Kiểm tra tính toàn vẹn dữ liệu trước khi huấn luyện:
```bash
python scripts/verify_d2sa_dataset.py
```

### 5.3. Huấn luyện & Đánh giá Cấu hình Chính (Row 4)
```bash
# Huấn luyện Row 4 trên D2SA (A100 GPU):
python scripts/train_d2sa.py \
    --epochs 30 \
    --batch-size 4 \
    --accumulation-steps 4 \
    --num-workers 8 \
    --val-subset-size 500 \
    --device cuda

# Đánh giá checkpoint sau 30 epoch:
python scripts/evaluate_d2sa.py \
    --checkpoint checkpoints/d2sa/amodal_shape_prediction_main_config/swin_amodal_epoch_30.pth \
    --device cuda
```

### 5.4. Chạy Toàn bộ 6 Cấu hình Ablation Study trên A100
Chạy toàn bộ 6 cấu hình tuần tự qua Bash script (hỗ trợ ngắt kết nối SSH với `nohup`):
```bash
# Cú pháp: bash scripts/run_ablation_d2sa.sh <device> <epochs> <batch_size> <acc_steps> <num_workers>
nohup bash scripts/run_ablation_d2sa.sh cuda 30 4 4 8 > ablation_runner.log 2>&1 &
echo $! > ablation_runner.pid
```
Hoặc chạy có chọn lọc bằng Python runner:
```bash
python scripts/run_ablation_d2sa.py --rows 4 5 6 --mode all --device cuda --epochs 30 --num-workers 8
```

---

## 🔬 6. Kiểm tra Trực quan Mặt nạ (Visual Sanity Checks)

Để đảm bảo không bị lỗi đảo trục $H/W$, lật ảnh (flip) hay lệch zero-padding trên các ảnh độ phân giải khác $1440 \times 1920$, 4 mẫu đại diện (tỉ lệ che $20\% - 60\%$) đã được kết xuất đầy đủ 4 ô:

1. **Mẫu 1 (`ethiquable_gruener_tee_ceylon`):** Kích thước gốc $1548 \times 1920$ (cần zero-pad), che $38.3\%$ $\to$ [sample_1.png](assets/d2sa_visual_checks/d2sa_visual_check_sample_1.png)
2. **Mẫu 2 (`kilimanjaro_tea_earl_grey`):** Kích thước gốc $1457 \times 1920$ (cần zero-pad), che $43.3\%$ $\to$ [sample_2.png](assets/d2sa_visual_checks/d2sa_visual_check_sample_2.png)
3. **Mẫu 3 (`cocoba_fruehstueckskakao_mit_honig`):** Kích thước $1440 \times 1920$, che $36.6\%$ $\to$ [sample_3.png](assets/d2sa_visual_checks/d2sa_visual_check_sample_3.png)
4. **Mẫu 4 (`gepa_bio_und_fair_kamillentee`):** Kích thước $1440 \times 1920$, che $32.5\%$ $\to$ [sample_4.png](assets/d2sa_visual_checks/d2sa_visual_check_sample_4.png)

---

## 📖 7. Tài liệu Kỹ thuật Chi tiết

- [scripts/README.md](scripts/README.md): Hướng dẫn chi tiết kỹ thuật, bảng delta tham số, dự toán ngoại suy thời gian và cơ chế resume.
- [a100_training_handoff_guide.md](a100_training_handoff_guide.md): Báo cáo nghiệm thu Phase 6, hướng dẫn bàn giao hệ thống huấn luyện A100 cho Thầy.