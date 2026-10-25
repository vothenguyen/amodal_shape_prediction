# 🧭 TÀI LIỆU BÀN GIAO TOÀN DIỆN DỰ ÁN CHO AI SUPERVISOR
# (COMPREHENSIVE PROJECT HANDOFF & SUPERVISOR BLUEPRINT)

> **Dự án:** Dự đoán biên dạng hoàn chỉnh của vật thể bị che khuất (**Amodal Instance Segmentation & Shape Prediction**)  
> **Kiến trúc cốt lõi:** Pipeline 2 giai đoạn (**SAM 2.1** Zero-shot + **Amodal Swin-UNet** 5 kênh)  
> **Tập dữ liệu chuẩn:** **COCOA** (COCO Amodal), **KINS** (KITTI In-house), **D2SA** (MVTec D2S Amodal)  
> **Mục tiêu của tài liệu này:** Cung cấp bức tranh toàn cảnh, chi tiết ở mức mã nguồn và toán học cho một **AI đóng vai trò Giám sát cấp cao (Supervisor / Technical Lead / Senior Research Scientist)**. AI Supervisor sẽ dựa vào tài liệu này để hiểu sâu sắc hiện trạng, từ đó thiết kế lộ trình nâng cấp (upgrade roadmap), sinh prompt thực thi chi tiết, và phản biện/định hướng kỹ thuật cho các giai đoạn phát triển tiếp theo của dự án.

---

## 📌 MỤC LỤC
1. [Tổng quan Dự án & Tình trạng Hiện tại (Executive Summary)](#1-tổng-quan-dự-án--tình-trạng-hiện-tại)
2. [Bài toán Khoa học & Kiến trúc Hai Giai đoạn (Problem Formulation & 2-Stage Pipeline)](#2-bài-toán-khoa-học--kiến-trúc-hai-giai-đoạn)
3. [Tổ chức Thư mục & Bóc tách Chức năng Từng File (Codebase Anatomy)](#3-tổ-chức-thư-mục--bóc-tách-chức-năng-từng-file)
4. [Nghiên cứu Thực nghiệm Ablation Study — Table V (Toán học & Code Mapping)](#4-nghiên-cứu-thực-nghiệm-ablation-study--table-v)
5. [Pipeline Dữ liệu & Tiền xử lý Đa Dataset (Data Engineering)](#5-pipeline-dữ-liệu--tiền-xử-lý-đa-dataset)
6. [Hạ tầng Huấn luyện Bền vững & Khả năng Chống Lỗi (Fault-Tolerant Infrastructure)](#6-hạ-tầng-huấn-luyện-bền-vững--khả-năng-chống-lỗi)
7. [Inference, Đánh giá Khoa học & Demo Web UI (Inference & Evaluation)](#7-inference-đánh-giá-khoa-học--demo-web-ui)
8. [Các Lộ trình Nâng cấp Tiềm năng (Strategic Upgrade Directions)](#8-các-lộ-trình-nâng-cấp-tiềm-năng)
9. [Bộ Quy tắc & Khung Sinh Prompt Dành Riêng cho AI Supervisor (Supervisor Prompting Framework)](#9-bộ-quy-tắc--khung-sinh-prompt-dành-riêng-cho-ai-supervisor)

---

## 1. TỔNG QUAN DỰ ÁN & TÌNH TRẠNG HIỆN TẠI

### 1.1. Đề tài Nghiên cứu
Trong thị giác máy tính truyền thống (*Modal Segmentation*), mô hình chỉ phân đoạn phần nhìn thấy được (*Visible Mask*) của vật thể. Khi các vật thể xếp chồng hoặc che khuất lẫn nhau (occlusion), thông tin hình học thực tế bị mất mát.  
Dự án này giải quyết bài toán **Amodal Instance Segmentation**: Tái tạo lại toàn bộ hình dáng nguyên vẹn (*Amodal Mask*) của đối tượng mục tiêu, đồng thời suy luận chính xác phần bị che khuất (*Occluded Region* $= Amodal \setminus Visible$).

### 1.2. Trạng thái Đạt được Hiện tại (Milestones Completed)
Dự án đã trải qua 6 giai đoạn (Phase 1 $\to$ Phase 6) và đạt được các cột mốc vững chắc:
1. **Hoàn thành trên tập COCOA:**
   - Đã huấn luyện và đánh giá đủ **6 cấu hình Ablation Study** (Row 1 $\to$ Row 6 theo Table V của nghiên cứu gốc), mỗi cấu hình chạy 30 epoch.
   - Script chuẩn nằm tại `scripts/model.py`, `scripts/train.py`, `scripts/evaluate.py`, và 5 biến thể tại `scripts/other_config_cocoa/`.
2. **Hoàn thành trên tập KINS:**
   - Đã điều chỉnh cho 8 lớp phương tiện giao thông, chạy đủ 6 cấu hình trên môi trường Google Colab qua notebook `scripts/train_kins.ipynb`.
3. **Mở rộng toàn diện sang tập D2SA (MVTec D2S Amodal):**
   - Đã xây dựng loader `scripts/dataset_d2sa.py` xử lý giải mã COCO RLE, zero-padding các ảnh có kích thước bất thường, gộp 2 split `training_rot0` và `augmented` thành 13,066 mẫu huấn luyện, và split `validation` gồm 15,654 mẫu.
   - Đồng bộ hoàn chỉnh **6 cấu hình Ablation cho D2SA** tại `scripts/train_d2sa.py` (Row 4) và `scripts/other_config_d2sa/` (Row 1, 2, 3, 5, 6).
   - Đã thực hiện **Smoke Test thành công trên máy local** (NVIDIA RTX 3050 4GB VRAM) cho cả 6 cấu hình, kiểm chứng gradient, shape, và lưu/đọc checkpoint.
   - Đã xây dựng bộ công cụ runner tự động: `scripts/run_ablation_d2sa.py` (Python) và `scripts/run_ablation_d2sa.sh` (Bash).
4. **Hệ thống Telemetry & Fault-Tolerance chuẩn Production:**
   - Logging 2 luồng đồng thời (`.log` người đọc, `.jsonl` telemetry máy đọc) kèm cơ chế `os.fsync()` tức thời.
   - Bắt tín hiệu ngắt `SIGTERM` từ Queue Manager (Slurm/PBS/Colab timeout) và lưu khẩn cấp trong $< 0.5\text{s}$.
   - Checkpoint ghi nguyên tử (`last.pth.tmp` $\to$ `os.replace` $\to$ `last.pth`) chống hỏng file khi bị `SIGKILL` hoặc đầy quota.
   - Thuật toán tua nhanh (`fast-forward`) learning rate scheduler khi resume với `--resume-epoch N`.
   - Giám sát cảnh báo sớm hiện tượng sụp đổ mô hình (*Model Collapse*) qua chỉ số `% Pixel Dương` (`PosPixels`).
5. **Giao diện Demo Tương tác Trực quan:**
   - Ứng dụng Gradio Web UI (`app.py`) tích hợp SAM 2.1 checkpoint `sam2.1_b.pt` và model Swin-UNet, hỗ trợ click chuột chọn đối tượng và chọn nhãn ngữ nghĩa để suy luận amodal thời gian thực.
6. **Sẵn sàng Huấn luyện A100:**
   - Toàn bộ tài liệu bàn giao máy chủ GPU mạnh đã đóng gói tại `a100_training_handoff_guide.md`. Mã nguồn sẵn sàng 100% để huấn luyện full 6 cấu hình trên A100.

---

## 2. BÀI TOÁN KHOA HỌC & KIẾN TRÚC HAI GIAI ĐOẠN

Hệ thống được thiết kế theo tư duy tách rời bài toán (*Decoupled Two-Stage Paradigm*):

```text
[Ảnh RGB I] ───► Stage 1: Phân đoạn cục bộ (SAM 2.1) ───► Visible Mask M_vis [H, W]
  + Point Prompt                                                   │
                                                                   ▼
                                             Toán tử hình thái (Dilate - Erode)
                                                                   │
                                                                   ▼
                                                          Edge Mask M_edge [H, W]
                                                                   │
┌──────────────────────────────────────────────────────────────────┘
▼
[Tensor 5 kênh: RGB (3ch) + Visible (1ch) + Edge (1ch)] ──► Tensor [B, 5, 224, 224]
                                                                   │
                                                                   ▼
                                                       Swin Transformer Encoder
                                                    (swin_tiny_patch4_window7_224)
                                                                   │
                                                                   ▼ (Bottleneck 768ch)
[Category ID c] ──► nn.Embedding(N, 768) ───────────────────► (+) Cộng vector ngữ nghĩa
                                                                   │
                                                                   ▼
                                                       U-Net Decoder (3 UpBlocks)
                                                                   │
                                                                   ▼ (Tùy chọn Spatial Attention)
                                                       Head Conv2d(64, 1, 1)
                                                                   │
                                                                   ▼
                                                        Amodal Mask Logits [B, 1, 224, 224]
                                                                   │
                                                                   ▼
                                     Hậu xử lý: Occluded Region = Amodal Mask \ Visible Mask
```

### 2.1. Stage 1 — Phân đoạn Cục bộ với SAM 2.1 (Local Segmentation)
- **Mô hình:** `Segment Anything Model 2.1 (SAM 2.1)` (bản `sam2.1_b.pt` hoặc `sam2.1_t.pt`).
- **Bản chất:** Foundation model zero-shot, **hoàn toàn đóng băng (frozen / không huấn luyện)** trong đồ án.
- **Đầu vào:** Ảnh RGB nguyên bản và tọa độ tương tác người dùng (Point prompt $(x, y)$ kèm label positive).
- **Đầu ra:** Mặt nạ nhìn thấy $M_{vis} \in \{0, 1\}^{H \times W}$.
- **Lợi ích:** Độc lập với dữ liệu gán nhãn amodal, tận dụng khả năng khái quát hóa cực cao của SAM 2.1 trên 1.1 tỷ mask.

### 2.2. Stage 2 — Suy luận Amodal với Swin-UNet (Amodal Reasoning)
- **Mô hình:** `AmodalSwinUNet` 5 kênh đầu vào kết hợp U-Net Decoder.
- **Kênh đầu vào (5 channels, $224 \times 224$):**
  1. `Channel 0..2`: Ảnh màu RGB chuẩn hóa.
  2. `Channel 3`: Mặt nạ nhìn thấy $M_{vis}$ (từ Stage 1 hoặc Ground-truth khi train).
  3. `Channel 4`: Mặt nạ đường biên $M_{edge} = \text{Dilate}(M_{vis}, K_5) - \text{Erode}(M_{vis}, K_5)$. Kênh này cung cấp *Inductive Bias* mạnh mẽ về ranh giới nhìn thấy của vật thể.
- **Backbone Encoder:** `swin_tiny_patch4_window7_224` (`timm`). Lớp tích chập Patch Embedding ban đầu $3 \to 96$ được thay thế bằng Conv2d $5 \to 96$, kích thước kernel $4 \times 4$, stride $4$:
  - 3 kênh đầu sao chép trọng số pretrained ImageNet.
  - 2 kênh mở rộng (Visible, Edge) được khởi tạo bằng 0 để giữ tính ổn định ban đầu.
- **Category Embedding:**
  - Nhãn lớp ($c \in [0, N-1]$) đi qua bảng tra `nn.Embedding(num_classes, 768)`.
  - Vector 768 chiều được broadcast và **cộng trực tiếp** vào feature map ở bottleneck của Swin Encoder ($B \times 768 \times 7 \times 7$). Giúp mô hình nắm bắt tiên nghiệm hình dạng (*shape prior*) theo từng loại vật thể (ví dụ: chai nước có dạng trụ đứng, xe hơi có mui vòm).
- **Decoder:** 3 khối `UpBlock` tích chập chuyển vị (`ConvTranspose2d` giảm 1/2 kênh, tăng kích thước $2\times$) kết hợp nối tắt (*skip connection*) từ encoder và khối `DoubleConv` (Conv3x3-BN-ReLU $\times 2$).
- **Spatial Attention Module (Cấu hình Row 5, Row 6):**
  - Trích xuất thông tin không gian qua phép gom kênh trung bình và lớn nhất: $[\text{AvgPool}(F); \text{MaxPool}(F)] \in \mathbb{R}^{B \times 2 \times H \times W}$.
  - Chiếu qua lớp tích chập Conv2d kernel $7 \times 7$, 2 kênh vào, 1 kênh ra, hàm kích hoạt Sigmoid tạo mặt nạ trọng số không gian.
- **Đầu ra:** Logits dự đoán mặt nạ amodal $M_{pred} \in \mathbb{R}^{B \times 1 \times 224 \times 224}$.

### 2.3. Hàm Mất mát Phạt Che Khuất (Occlusion-Aware Loss)
Vùng nhìn thấy $M_{vis}$ thường chiếm phần lớn diện tích, trong khi vùng bị che khuất $M_{occ} = M_{amodal} \setminus M_{vis}$ chiếm tỷ lệ nhỏ, dẫn đến hiện tượng mất cân bằng điểm ảnh trầm trọng. Hàm mất mát được thiết kế đặc thù:
$$\mathcal{L} = \mathcal{L}_{\text{Occ-BCE}} + \mathcal{L}_{\text{SoftDice}}$$
Trong đó:
$$\mathcal{L}_{\text{Occ-BCE}} = -\frac{1}{|\Omega|} \sum_{i \in \Omega} w_i \cdot \left[ y_i \log(\sigma(\hat{y}_i)) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right]$$
Với trọng số điểm ảnh $w_i$:
$$w_i = \begin{cases} 
5.0 & \text{nếu } i \in M_{occ} \quad (y_i = 1 \land M_{vis, i} = 0) \\ 
1.0 & \text{ngược lại} 
\end{cases}$$
Phạt gấp **5 lần** sai số trên vùng bị che khuất, ép mô hình phải tập trung tái tạo phần bị che thay vì chỉ học sao chép lại vùng nhìn thấy.

---

## 3. TỔ CHỨC THƯ MỤC & BÓC TÁCH CHỨC NĂNG TỪNG FILE

```text
amodal_shape_prediction-main/
├── app.py                             # [UI] Ứng dụng Gradio Web demo tương tác 2-stage
├── requirements.txt                   # [ENV] Danh sách thư viện Python và phiên bản
├── README.md                          # [DOC] Tài liệu tổng quan dự án
├── a100_training_handoff_guide.md     # [DOC] Báo cáo Phase 6 & cẩm nang huấn luyện máy chủ A100
├── ANTIGRAVITY_AGENT_PROMPT.md        # [DOC] Prompt lịch sử điều phối Phase 1-6
├── PROMPT_LOGGING_SYSTEM.md           # [DOC] Prompt đặc tả hệ thống logging & fault tolerance
├── .gitattributes                     # [CONFIG] Chuẩn hóa ngắt dòng LF cho scripts Bash
├── .gitignore                         # [CONFIG] Loại trừ data/, checkpoints/, logs/, results/
├── assets/                            # [MEDIA] Hình ảnh tài liệu & kiểm chứng trực quan
│   ├── Table.png                      # Bảng Table V gốc trong paper
│   ├── figures/                       # Sơ đồ khối kiến trúc
│   └── d2sa_visual_checks/            # Ảnh kiểm tra 4 ô hình học và mask D2SA
├── checkpoints/                       # [MODEL] Chứa checkpoint .pth (tự sinh khi train)
│   └── d2sa/                          # Checkpoint phân chia theo từng cấu hình D2SA
├── data/                              # [DATA] Dữ liệu benchmark (COCOA, KINS, D2SA)
├── docs/                              # [DOCS] Hệ thống tài liệu phân tích chuyên sâu
│   ├── PROJECT_UNDERSTANDING.md       # Tổng kết Phase 1 (đọc hiểu codebase)
│   ├── PROPOSED_METHOD.md             # Chi tiết phương pháp toán học và kiến trúc
│   ├── DATASET_INSTANCES_DETAILED.md  # Thống kê chi tiết instance các split D2SA
│   ├── TRAINING_SUMMARY.md            # Tóm tắt quá trình huấn luyện
│   ├── guides/                        # Hướng dẫn setup môi trường Colab, Quick reference
│   ├── occlusion/                     # Phân tích chiến lược trọng số occlusion
│   └── reports/                       # Báo cáo kỹ thuật chi tiết
├── logs/                              # [LOGS] Nhật ký huấn luyện đa luồng (.log & .jsonl)
├── results/                           # [METRICS] Kết quả benchmark định lượng dạng JSON
├── scratch/                           # [TEMP] File nháp thử nghiệm
└── scripts/                           # [CORE CODE] Mã nguồn huấn luyện, đánh giá & ablation
    ├── model.py                       # Kiến trúc Swin-UNet chuẩn (Row 4 COCOA/D2SA)
    ├── dataset.py                     # Dataset loader cho COCOA (đọc polygon, order)
    ├── train.py                       # Script train Row 4 trên COCOA
    ├── evaluate.py                    # Script đánh giá single-image trên COCOA
    ├── train_lambda.py                # Thử nghiệm so sánh các trọng số lambda occlusion
    ├── train_kins.ipynb               # Notebook Colab huấn luyện 6 cấu hình trên KINS
    ├── dataset_d2sa.py                # Dataset loader chuyên biệt cho D2SA (RLE, padding)
    ├── train_d2sa.py                  # Script train Row 4 (Main Config) trên D2SA
    ├── evaluate_d2sa.py               # Script benchmark chuyên biệt cho D2SA
    ├── verify_d2sa_dataset.py         # Công cụ kiểm tra toàn vẹn dataset D2SA trước khi train
    ├── logging_utils.py               # Thư viện logging nguyên tử, SIGTERM handler, telemetry
    ├── run_ablation_d2sa.py           # Bộ điều phối tự động 6 cấu hình D2SA bằng Python
    ├── run_ablation_d2sa.sh           # Script Bash chạy nền/Slurm trên cụm máy chủ Linux
    ├── README.md                      # Hướng dẫn chi tiết Table V và lệnh chạy ablation
    ├── other_config_cocoa/            # 5 cấu hình ablation trên COCOA (Row 1, 2, 3, 5, 6)
    └── other_config_d2sa/             # 5 cấu hình ablation trên D2SA (Row 1, 2, 3, 5, 6)
```

### Bóc tách Chi tiết các File Cốt lõi (`scripts/`)

| Tên File | Vai trò & Trách nhiệm Kỹ thuật | Input / Output |
|:---|:---|:---|
| [`scripts/model.py`](scripts/model.py) | Định nghĩa lớp `AmodalSwinUNet` (Row 4). Tích hợp Swin Tiny, mở rộng Patch Embed từ 3 lên 5 kênh, nhúng Category Embedding 768 chiều vào bottleneck, U-Net Decoder 3 cấp. | In: Tensor $[B, 5, 224, 224]$, Class IDs $[B]$.<br>Out: Logits $[B, 1, 224, 224]$. |
| [`scripts/dataset_d2sa.py`](scripts/dataset_d2sa.py) | Lớp `AmodalDatasetD2SA` và `AmodalDatasetD2SA_Concat`. Giải mã mặt nạ COCO RLE, xử lý zero-pad các ảnh lệch chuẩn, sinh $M_{edge}$ qua phép co/giãn hình thái, áp dụng Albumentations đồng bộ trên ảnh và mask. | In: File ảnh `.jpg`, File JSON annotation D2SA.<br>Out: `(input_5ch, amodal_mask, occluded_mask, class_id)`. |
| [`scripts/train_d2sa.py`](scripts/train_d2sa.py) | Điểm vào huấn luyện cấu hình chính (Row 4) trên D2SA. Tích hợp AdamW, CosineAnnealingLR, Gradient Accumulation, Dual-level validation, Atomic Checkpoint, và Resume. | In: Tham số CLI (`--epochs`, `--batch-size`, `--lr`, `--resume-epoch`...).<br>Out: Checkpoint `.pth`, file log `.log`, `.jsonl`. |
| [`scripts/evaluate_d2sa.py`](scripts/evaluate_d2sa.py) | Đánh giá checkpoint trên 15,654 mẫu validation của D2SA. Tính toán toàn diện: **mIoU**, **Dice**, **Precision**, **Recall**, và **Invisible-mIoU** (đo đạc riêng trên vùng bị che). | In: Đường dẫn `--checkpoint`.<br>Out: Kết quả in console và lưu JSON tại `results/`. |
| [`scripts/logging_utils.py`](scripts/logging_utils.py) | Module chịu lỗi và telemetry: `FlushFileHandler` (gọi `os.fsync`), `SafeStreamHandler` (chống lỗi Unicode cp1252), cơ chế bắt tín hiệu `SIGTERM`, và hàm `atomic_save_checkpoint`. | Cung cấp đối tượng logger, hook cứu hộ và cơ chế ghi file an toàn. |
| [`scripts/verify_d2sa_dataset.py`](scripts/verify_d2sa_dataset.py) | Sanity check dữ liệu trước huấn luyện: đếm 22,562 ảnh, kiểm tra tính hợp lệ của 3 file JSON, kiểm tra xung đột `image_id`, và chạy thử loader đọc ngẫu nhiên. | In: Thư mục `data/D2SA`.<br>Out: Báo cáo tính toàn vẹn (Pass/Fail). |
| [`scripts/run_ablation_d2sa.py`](scripts/run_ablation_d2sa.py) | Điều phối chạy tuần tự hoặc song song các hàng ablation (từ 1 đến 6). Chuyển tiếp toàn bộ cờ CLI (resume, workers, accumulation...). | In: Đối số `--rows 1,2,3,4,5,6`.<br>Thực thi các script con tương ứng. |
| [`app.py`](app.py) | Ứng dụng Gradio Web UI tương tác trực tiếp. Nạp SAM 2.1 từ `ultralytics` và `AmodalSwinUNet`, cho phép click chuột lấy điểm mốc, suy luận và vẽ overlay trực quan 3 màu. | Giao diện Web trên trình duyệt (`http://127.0.0.1:7860`). |

---

## 4. NGHIÊN CỨU THỰC NGHIỆM ABLATION STUDY — TABLE V

Nghiên cứu bám sát chặt chẽ **Table V** từ công trình tham chiếu, khảo sát vai trò của 4 thành phần: **Edge Mask**, **Category Embedding**, **Spatial Attention**, và **Hàm mất mát Occlusion-Aware**.  
Tất cả các script đều chứa lệnh `assert total_params == EXPECTED_PARAMS` khi khởi động nhằm loại bỏ 100% rủi ro cấu hình sai lệch.

### 4.1. Bảng Đối chiếu 6 Cấu hình (Table V Mapping)

| Row | Tên cấu hình / Đường dẫn thư mục | Kênh vào | Edge Mask | Cat Embedding | Spatial Attention | Hàm mất mát (Loss) | Số tham số thực tế | Forward Signature |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/` | **4** (RGB+Vis) | ❌ | ❌ | ❌ | BCEWithLogits | **34,352,027** | `forward(x)` |
| **2** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/` | **5** (RGB+Vis+Edge) | ✅ | ❌ | ❌ | BCEWithLogits | **34,353,563** | `forward(x)` |
| **3** | `other_config_d2sa/amodal-shape-prediction-no-spatial-no-embeding/` | **5** (RGB+Vis+Edge) | ✅ | ❌ | ❌ | Occ-Aware (5x) + Dice | **34,353,563** | `forward(x)` |
| **4** | **Cấu hình chính (Main / Baseline)**<br>• D2SA: `scripts/train_d2sa.py` | **5** (RGB+Vis+Edge) | ✅ | ✅ (60 lớp) | ❌ | Occ-Aware (5x) + Dice | **34,399,643** | `forward(x, class_ids)` |
| **5** | `other_config_d2sa/amodal_shape_prediction_no_embeding/` | **5** (RGB+Vis+Edge) | ✅ | ❌ | ✅ ($k=7$) | Occ-Aware (5x) + Dice | **34,353,661** | `forward(x)` |
| **6** | `other_config_d2sa/amodal_shape_prediction_full_config/` | **5** (RGB+Vis+Edge) | ✅ | ✅ (60 lớp) | ✅ ($k=7$) | Occ-Aware (5x) + Dice | **34,399,741** | `forward(x, class_ids)` |

### 4.2. Chứng minh Toán học về Quan hệ Tham số giữa các Hàng
1. **Row 2 so với Row 1 ($\Delta = +1,536$ tham số):**  
   Patch Embedding của Swin Transformer nhận kích thước patch $4 \times 4$, số chiều đặc trưng $96$. Khi bổ sung thêm 1 kênh (Edge Mask):
   $$\Delta_{\text{Row2-Row1}} = 96 \times 1 \times 4 \times 4 = 1,536 \implies 34,352,027 + 1,536 = 34,353,563$$
2. **Row 3 so với Row 2 ($\Delta = 0$ tham số):**  
   Kiến trúc mạng giữ nguyên tuyệt đối, chỉ thay đổi hàm mục tiêu từ BCE tiêu chuẩn sang hàm Occlusion-Aware Loss kết hợp Soft Dice.
3. **Row 4 so với Row 3 ($\Delta = +46,080$ tham số):**  
   Bổ sung bảng nhúng ngữ nghĩa `nn.Embedding(60, 768)` cho 60 lớp đối tượng của D2SA vào bottleneck:
   $$\Delta_{\text{Row4-Row3}} = 60 \times 768 = 46,080 \implies 34,353,563 + 46,080 = 34,399,643$$
   *(Ghi chú: Trên tập COCOA với 91 lớp, delta tương ứng là $91 \times 768 = 69,888$ tham số).*
4. **Row 5 so với Row 3 ($\Delta = +98$ tham số):**  
   Bổ sung khối Spatial Attention (kết hợp AvgPool và MaxPool dọc trục kênh qua một lớp tích chập Conv2d kernel $7 \times 7$, 2 kênh vào, 1 kênh ra, không dùng bias):
   $$\Delta_{\text{Row5-Row3}} = 1 \times 2 \times 7 \times 7 = 98 \implies 34,353,563 + 98 = 34,353,661$$
5. **Row 6 so với Row 4 ($\Delta = +98$ tham số):**  
   Bổ sung khối Spatial Attention ($98$ tham số) trên cấu hình Row 4 đã có Category Embedding.
   $$\Delta_{\text{Row6-Row4}} = 98 \implies 34,399,643 + 98 = 34,399,741$$

---

## 5. PIPELINE DỮ LIỆU & TIỀN XỬ LÝ ĐA DATASET

Dự án hỗ trợ chuẩn hóa trên 3 tập dữ liệu có đặc thù định dạng khác nhau:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          3 DATASET BENCHMARK                           │
├─────────────────────┬──────────────────────┬───────────────────────────┤
│    COCOA (COCO)     │     KINS (KITTI)     │       D2SA (MVTec)        │
├─────────────────────┼──────────────────────┼───────────────────────────┤
│ • 91 lớp tự nhiên   │ • 8 lớp giao thông   │ • 60 lớp sản phẩm grocery │
│ • Đọc đa giác       │ • Bbox + Polygon     │ • Giải mã RLE Mask        │
│   (Polygon regions) │ • Xe cộ, người đi bộ │ • Ảnh độ phân giải lớn    │
│ • Suy ra Visible    │ • Suy diễn ngữ cảnh  │   1440x1920 (cần zero-pad)│
│   qua trường order  │   đường phố          │ • Gộp train + augmented   │
└─────────────────────┴──────────────────────┴───────────────────────────┘
```

### 5.1. Đặc thù Tiền xử lý D2SA
1. **Giải mã RLE (Run-Length Encoding):** Annotation của D2SA lưu cả 2 trường `segmentation` (Amodal mask) và `visible_mask` (Visible mask) dạng RLE nén. Dataset loader sử dụng `pycocotools.mask.decode()` để giải nén thành mảng nhị phân numpy uint8.
2. **Kỹ thuật Zero-Padding chống méo hình học:**  
   Khoảng 5% số ảnh D2SA có vật thể tràn biên được lưu với kích thước thực tế lệch chuẩn (ví dụ $1548 \times 1920$ hoặc $1457 \times 1920$). Dataset loader tự động phát hiện và pad thêm viền đen về đúng tỷ lệ chuẩn trước khi đưa qua Albumentations resize về $224 \times 224$, bảo toàn 100% tỷ lệ hình học của vật thể.
3. **Phân tách `image_id` an toàn tuyệt đối:**  
   Tập huấn luyện là sự kết hợp giữa `D2S_amodal_training_rot0.json` (438 ảnh, 690 instances, IDs 200..44,520) và `D2S_amodal_augmented.json` (1,562 ảnh, 12,376 instances, IDs 88,000,000..88,001,561). Hai tập có dải ID phân tách rời rạc, giao nhau bằng 0, đảm bảo tính toàn vẹn khi gộp.
4. **Data Augmentation:**  
   Sử dụng Albumentations với pipeline đồng bộ:
   ```python
   A.Compose([
       A.Resize(224, 224),
       A.HorizontalFlip(p=0.5),
       A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
       A.RandomBrightnessContrast(p=0.2),
   ], additional_targets={'visible_mask': 'mask', 'occluded_mask': 'mask'})
   ```

---

## 6. HẠ TẦNG HUẤN LUYỆN BỀN VỮNG & KHẢ NĂNG CHỐNG LỖI

Hệ thống được thiết kế đặc thù cho các môi trường điện toán đám mây hoặc máy chủ cụm (Slurm/PBS/Colab) nơi tiến trình có thể bị chấm dứt bất kỳ lúc nào vì hết quota hoặc hết thời gian gán slot GPU.

### 6.1. Chiến lược Checkpoint Hai Tầng & Ghi Nguyên Tử (Atomic Write)
Nếu lưu toàn bộ trạng thái AdamW (405 MB/checkpoint) cho 30 epoch $\times$ 6 cấu hình sẽ chiếm hơn **73 GB**, dễ làm đầy ổ đĩa server trường. Hệ thống áp dụng cơ chế 2 tầng:
1. **Checkpoint Định kỳ (`swin_amodal_epoch_X.pth`):** Chỉ lưu `model.state_dict()` (~131 MB) phục vụ đánh giá sau khi hoàn tất training.
2. **Checkpoint Khôi phục (`last.pth`):** Lưu trọn vẹn `model_state_dict`, `optimizer_state_dict` (moment $m_t, v_t$ của AdamW), `scheduler_state_dict`, và metadata `epoch`. File này được ghi đè nguyên tử mỗi epoch:
   - Dữ liệu ghi ra file tạm `last.pth.tmp`.
   - Gọi `file.flush()` và `os.fsync(file.fileno())` ép dữ liệu xuống phiến đĩa.
   - Dùng `os.replace("last.pth.tmp", "last.pth")`.  
   *Ý nghĩa:* Loại trừ 100% khả năng file checkpoint bị corrupt nếu tiến trình bị `SIGKILL` giữa chừng. Dung lượng đĩa toàn dự án được khống chế ở mức an toàn **~26 GB**.

### 6.2. Cứu hộ Tín hiệu `SIGTERM`
Các queue manager như Slurm thường gửi tín hiệu `SIGTERM` trước khi gửi `SIGKILL` cứng khoảng 30–60 giây.  
`logging_utils.py` đăng ký hook xử lý tín hiệu: ngay khi nhận `SIGTERM`, tiến trình lập tức đóng gói toàn bộ trạng thái hiện tại thành `emergency_checkpoint_epoch_X.pth` trong thời gian $< 0.5\text{s}$, flush toàn bộ log và thoát an toàn.

### 6.3. Thuật toán Resume & Fast-Forward Scheduler
Khi resume với `--resume-epoch N`:
- Mô hình nạp trọng số và trạng thái optimizer từ epoch $N$.
- Tránh lỗi phổ biến của PyTorch `KeyError: param 'initial_lr'`: Thuật toán chạy vòng lặp tua nhanh `scheduler.step()` đúng $N$ bước với cảnh báo bị tắt.
- Learning rate tại epoch $N+1$ khớp chính xác đến từng số thập phân trên đường cong `CosineAnnealingLR` (độ lệch $< 10^{-11}$).

### 6.4. Đánh giá Hai Tầng (Dual-Level Validation)
Tập validation của D2SA có tới 15,654 mẫu. Một lượt đánh giá đầy đủ mất khoảng 5–6 phút trên A100:
- **Epoch 5, 10, 15, 20, 25:** Tự động đánh giá trên tập con `--val-subset-size 500` (mất ~10 giây) để giám sát sự hội tụ và đo `% Pixel Dương`.
- **Epoch 30 (Epoch cuối):** Tự động chuyển sang đánh giá toàn bộ **15,654 mẫu** để lấy số liệu chính xác phục vụ báo cáo khoa học.  
*Tiết kiệm xấp xỉ 3 giờ GPU cho mỗi lượt chạy full 6 cấu hình.*

### 6.5. Cảnh báo Sớm Sụp đổ Mô hình (Model Collapse Detection)
Tích hợp metric `% Pixel Dương` (`PosPixels` / `pos_pixel_pct`):
$$\text{PosPixels} = \frac{\sum \mathbb{I}(M_{pred} > 0.5)}{H \times W \times B} \times 100\%$$
- Nếu $\text{PosPixels} \to 0.0\%$: Mô hình sụp đổ về nền đen (all-background).
- Nếu $\text{PosPixels} \to 100.0\%$: Mô hình sụp đổ về toàn bộ dương tính (all-foreground).
- Ngưỡng bình thường của D2SA: **$8.0\% - 20.0\%$**. Giúp phát hiện lỗi bùng nổ gradient ngay từ epoch đầu tiên.

---

## 7. INFERENCE, ĐÁNH GIÁ KHOA HỌC & DEMO WEB UI

### 7.1. Giao diện Trực quan Gradio (`app.py`)
- Cho phép người dùng tải lên ảnh bất kỳ, click tương tác chuột chọn đối tượng mục tiêu.
- Chọn nhãn ngữ nghĩa từ 90 lớp đối tượng COCO.
- Pipeline kích hoạt SAM 2.1 trích xuất $M_{vis}$ $\to$ Amodal Swin-UNet dự đoán $M_{amodal}$ $\to$ Trực quan hóa 3 vùng:
  - *Màu xanh lá:* Phần nhìn thấy ($M_{vis}$).
  - *Màu đỏ:* Biên dạng hoàn chỉnh ($M_{amodal}$).
  - *Màu xanh dương:* Vùng bị che khuất ($M_{occ} = M_{amodal} \setminus M_{vis}$).

### 7.2. Bộ Chỉ số Đánh giá Định lượng
Được tính toán độc lập trên từng đối tượng và lấy trung bình toàn tập (Single-image Instance-level Evaluation):
1. **Mean Intersection-over-Union (mIoU):** $\frac{|M_{pred} \cap M_{gt}|}{|M_{pred} \cup M_{gt}|}$
2. **Dice Similarity Coefficient (Dice):** $\frac{2 |M_{pred} \cap M_{gt}|}{|M_{pred}| + |M_{gt}|}$
3. **Precision & Recall:** Đo lường độ chính xác và độ bao phủ điểm ảnh.
4. **Invisible-mIoU (Inv-mIoU):** Đo lường riêng trên vùng bị che khuất ($M_{occ} = M_{gt} \setminus M_{vis}$). Đây là chỉ số phản ánh năng lực cốt lõi của bài toán amodal.

### 7.3. Quy chuẩn Báo cáo Khoa học (Scientific Protocol)
> 🎯 **NGUYÊN TẮC BẮT BUỘC:**  
> 1. Tập Test của D2SA không có nhãn công khai, do đó tập Validation (15,654 mẫu) được dùng làm tập benchmark chính thức (theo chuẩn của các paper VRSP-Net, AISFormer).  
> 2. **Bắt buộc báo cáo kết quả tại Epoch 30 cố định.** Tuyệt đối không chọn "Best Epoch trên validation" nhằm tránh thiên kiến lạc quan (*optimistic bias*) và rò rỉ dữ liệu (*data leakage*), đảm bảo tính công bằng tuyệt đối khi đối chiếu với COCOA và KINS.

---

## 8. CÁC LỘ TRÌNH NÂNG CẤP TIỀM NĂNG (STRATEGIC UPGRADE DIRECTIONS)

Dưới góc nhìn của một **AI Supervisor**, đây là các hướng nghiên cứu và nâng cấp giá trị cao mà AI có thể định hướng và tạo prompt cho developer agent thực thi:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                     CÁC TRỤC NÂNG CẤP TIỀM NĂNG                        │
├────────────────────────┬───────────────────────┬───────────────────────┤
│ 1. Kiến trúc Mô hình   │ 2. Pipeline Tự động   │ 3. Triển khai & Demo  │
├────────────────────────┼───────────────────────┼───────────────────────┤
│ • SAM 2 End-to-End     │ • Thay click chuột    │ • Nâng cấp Gradio Pro │
│   Adapter Tuning       │   bằng YOLOv8-seg /   │   Web UI cao cấp      │
│ • Backbone Swin-B/L    │   Grounding DINO      │ • Tối ưu hóa TensorRT │
│ • Diffusion Amodal     │ • Tự động phát hiện   │   ONNX, FP16/INT8     │
│   Completion           │   và suy luận toàn    │ • Video Amodal        │
│ • Mask2Former Query    │   bộ cảnh amodal      │   Tracking thời       │
│   Head                 │   (Fully Autonomous)  │   gian thực           │
└────────────────────────┴───────────────────────┴───────────────────────┘
```

### Hướng 1: Nâng cấp Kiến trúc Mô hình (Model Architecture Upgrade)
- **SAM 2 Adapter / LoRA End-to-End Fine-tuning:** Thay vì đóng băng hoàn toàn SAM 2.1 và dùng Swin-UNet riêng biệt ở Stage 2, tích hợp các module LoRA hoặc Adapter trực tiếp vào Mask Decoder của SAM 2.1 để dự đoán trực tiếp amodal mask và occlusion mask trong một lần forward duy nhất.
- **Nâng cấp Backbone:** Nâng cấp từ `swin_tiny` ($34.4\text{M}$ tham số) lên `swin_base` hoặc `swin_large` để tăng năng lực trích xuất đặc trưng không gian.
- **Tích hợp Transformer Queries (Mask2Former / AISFormer style):** Sử dụng các amodal queries có thể học (*learnable amodal queries*) tương tác chéo (*cross-attention*) với đặc trưng ảnh và prompt visual tokens.
- **Generative Diffusion Amodal Inpainting:** Tận dụng mô hình khuếch tán (Latent Diffusion) để sinh hình thái vật thể bị che khuất với độ chân thực cao đối với các ca che khuất nghiêm trọng ($> 80\%$).

### Hướng 2: Chuyển đổi Pipeline Tự động Hoàn toàn (Fully Autonomous Pipeline)
- Hiện tại Stage 1 yêu cầu người dùng click chuột tương tác (Point prompt) hoặc lấy ground-truth visible mask.
- **Nâng cấp:** Tích hợp bộ phát hiện đối tượng tự động ở đầu vào:
  - Sử dụng **YOLOv8-seg / YOLO11-seg** hoặc **Mask R-CNN** để sinh tự động tất cả các bounding box và visible mask của toàn bộ vật thể trong ảnh.
  - Sử dụng **Grounding DINO** cho phép người dùng nhập text prompt (ví dụ: "find all occluded cereal boxes") để tự động suy luận amodal.
  - Xây dựng sơ đồ phân cấp thứ tự che khuất (*Occlusion Order Graph / Depth Sorting*) để giải quyết quan hệ tương hỗ giữa nhiều vật thể đè lên nhau.

### Hướng 3: Nâng cấp Kỹ thuật Dữ liệu & Học Đa Nhiệm (Data & Multi-Task)
- **Tổng hợp Che khuất Nhân tạo (Synthetic Occlusion Augmentation):** Tự động cắt các vật thể từ ảnh khác đè lên vật thể mục tiêu trong tập huấn luyện để mô phỏng đa dạng mức độ che khuất (từ 10% đến 90%).
- **Multi-task Learning:** Huấn luyện mô hình đồng thời dự đoán: (1) Amodal Mask, (2) Occluded Mask, (3) Bản đồ độ sâu tương đối (*Relative Depth / Occlusion Boundary Map*).

### Hướng 4: Triển khai & Trải nghiệm Người dùng (Deployment & UI/UX)
- **Nâng cấp Gradio Web UI thành ứng dụng cao cấp:**
  - Hỗ trợ xem trực quan trên cả 3 tập dữ liệu (COCOA, KINS, D2SA).
  - Thêm tính năng so sánh trực tiếp song song (Side-by-side) giữa các mô hình ablation (ví dụ: Row 1 vs Row 4 vs Row 6).
  - Tải lên video và thực hiện amodal tracking qua từng frame.
- **Tối ưu hóa Tốc độ Suy luận (Production Inference):** Export mô hình sang ONNX Runtime hoặc TensorRT FP16, giảm độ trễ từ ~40ms xuống < 10ms.

---

## 9. BỘ QUY TẮC & KHUNG SINH PROMPT DÀNH RIÊNG CHO AI SUPERVISOR

Khi bạn (AI thứ hai) đảm nhận vai trò **Supervisor**, bạn cần hành xử như một **Giám đốc Kỹ thuật / Trưởng nhóm Nghiên cứu (Chief AI Architect / Principal Scientist)**. Dưới đây là các chỉ dẫn cốt tử:

### 9.1. Triết lý Điều hành (Core Principles)
1. **Tuân thủ Triết lý Thực dụng (Pragmatic Engineering — theo `.clinerules`):**
   - *"The best code is the code you never wrote"*: Không thêm abstraction thừa, không tạo class/file nếu có thể giải quyết sạch sẽ bằng thư viện chuẩn hoặc tái sử dụng code hiện có.
   - Luôn phản biện yêu cầu (*Push back first*): Nếu một ý tưởng nâng cấp quá phức tạp so với giá trị khoa học mang lại, hãy điều chỉnh hoặc tinh giản trước khi ra lệnh viết code.
   - Tái sử dụng triệt để: Mọi logic logging, data loading, checkpointing đã được viết cực kỳ chuẩn mực trong `scripts/logging_utils.py` và `scripts/dataset_d2sa.py`. Hãy tái sử dụng, không viết lại từ đầu.
2. **Kỷ luật Báo cáo Khoa học (Scientific Rigor):**
   - Mọi thực nghiệm phải giữ nguyên bộ siêu tham số đối chuẩn (AdamW 1e-4, CosineAnnealing 30 epochs) trừ khi mục tiêu nghiên cứu là tuning hyperparameter.
   - Bắt buộc kiểm chứng bằng assertion số lượng tham số hoặc delta tham số.
   - Không được thay đổi protocol đánh giá: Luôn evaluate trên tập validation chuẩn ở Epoch 30.
3. **Phân rã Công việc theo Phase Tuần tự (Phased Execution):**
   - Tuyệt đối không yêu cầu developer agent làm tất cả trong một prompt duy nhất. Luôn chia nhỏ thành:
     - *Phase 1:* Khảo sát & Phân tích tác động.
     - *Phase 2:* Tái cấu trúc / Chuẩn bị môi trường.
     - *Phase 3:* Triển khai mã nguồn lõi.
     - *Phase 4:* Smoke test trên máy local (RTX 3050 / VRAM nhỏ / subset mẫu).
     - *Phase 5:* Đóng gói & Viết tài liệu kiểm chứng.

### 9.2. Mẫu Cấu trúc Prompt Chuẩn (Supervisor Prompt Template)
Khi AI Supervisor tạo prompt cho Developer Agent, hãy tuân theo khung chuẩn sau:

```markdown
# 🎯 PROMPT CHO DEVELOPER AGENT: [TÊN TÍNH NĂNG / NHIỆM VỤ NÂNG CẤP]

## 1. MỤC TIÊU & BỐI CẢNH (CONTEXT)
- **Nhiệm vụ:** [Mô tả mục tiêu kỹ thuật ngắn gọn]
- **Tập tin liên quan:** [Liệt kê các file cần sửa hoặc tạo mới, ví dụ: scripts/model.py, app.py...]
- **Nguyên tắc kỹ thuật:** Bám sát PROJECT_SUPERVISOR_HANDOFF.md, tái sử dụng logging_utils.py, không phá vỡ logic cũ của COCOA/KINS/D2SA.

## 2. YÊU CẦU KỸ THUẬT CHI TIẾT (SPECIFICATIONS)
- [Mô tả chi tiết kiến trúc, tensor shape, hàm mất mát, hoặc giao diện UI cần thêm]
- [Ràng buộc về VRAM: Giữ batch size phù hợp cho máy local 4GB khi test]
- [Ràng buộc về số lượng tham số: Tính toán rõ delta tham số nếu có thêm module mới]

## 3. CÁC BƯỚC THỰC HIỆN TUẦN TỰ (PHASED TASKS)
- **Bước 1 (Kiểm tra & Chuẩn bị):** Đọc kỹ các file liên quan trước khi sửa...
- **Bước 2 (Viết code lõi):** Triển khai tại file X...
- **Bước 3 (Smoke Test cục bộ):** Chạy lệnh thử nghiệm trên subset nhỏ hoặc 1 epoch, kiểm tra exit code = 0...
- **Bước 4 (Kiểm tra chất lượng):** Xác nhận không có lỗi NaN, lỗi rò rỉ bộ nhớ, hoặc crash log...

## 4. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)
- [ ] Code tuân thủ coding style hiện tại, có docstring tiếng Việt/Anh chuẩn.
- [ ] Kiểm thử chạy thành công trên máy local mà không bị Out-Of-Memory (OOM).
- [ ] Checkpoint lưu đầy đủ model + optimizer + scheduler.
- [ ] Cập nhật báo cáo tóm tắt thay đổi.
```

---

## 10. TỔNG KẾT

Tài liệu này là **kim chỉ nam toàn diện nhất** của dự án `amodal_shape_prediction`. Mọi quyết định thiết kế, từ cách bóc tách từng kênh đầu vào $5\text{ch}$, cách cộng vector nhúng ngữ nghĩa $768\text{d}$, cách xử lý zero-padding trên ảnh D2SA, cho đến cơ chế ghi đĩa nguyên tử `atomic_save_checkpoint` đều được ghi lại với độ chi tiết tối đa.

**AI Supervisor có thể tự tin sử dụng tài liệu này để định hướng, thiết kế các nghiên cứu nâng cao, và đồng hành cùng nhóm phát triển đưa dự án lên tầm vóc bài báo khoa học chất lượng cao.**
