# 👁️ Amodal Shape Prediction (Swin-UNet + SAM 2.1)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue.svg?style=flat-square&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-UI-blue.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)

Báo cáo dự án Nghiên cứu Khoa học / Đồ án tốt nghiệp: **Amodal Shape Prediction** — Dự đoán hình dạng đầy đủ của vật thể ngay cả khi bị che khuất (occlusion).

Hệ thống được thiết kế theo pipeline 2 giai đoạn:
1. **Stage 1 (Phân đoạn cục bộ):** `SAM 2.1` trích xuất mặt nạ hiển thị (Visible Mask) từ điểm click chuột (zero-shot).
2. **Stage 2 (Suy luận Amodal):** `Amodal Swin-UNet` đa kênh kết hợp Category Embedding và hàm mất mát Occlusion-Aware Loss để tái tạo toàn bộ hình dáng (Amodal Mask).

---

## 🚀 Hướng Dẫn Chạy Trên Google Colab A100 (40GB/80GB VRAM)

Toàn bộ quá trình chuẩn bị dữ liệu, chạy huấn luyện 1 lần đầy đủ 6 cấu hình (Ablation Study) và tự động lưu trực tiếp vào Google Drive được đóng gói tại:

👉 **[COLAB_A100_GUIDE.md](COLAB_A100_GUIDE.md)**: Cẩm nang hướng dẫn từng bước chi tiết.  
👉 **[train_d2sa_colab_a100.ipynb](train_d2sa_colab_a100.ipynb)**: Jupyter Notebook sẵn sàng chạy 1-click trên Colab.

---

## 🧠 Kiến Trúc Mô Hình (Amodal Swin-UNet)

- **Đầu vào (Input Tensor [B, 5, 224, 224]):**
  1. `Kênh 1-3:` Ảnh RGB gốc.
  2. `Kênh 4:` Visible Mask (phần nhìn thấy được).
  3. `Kênh 5:` Edge Mask (ranh giới bị che khuất, sinh bằng phép giãn/xói mòn hình thái học).
- **Encoder:** Backbone Swin Transformer (`swin_tiny_patch4_window7_224`, pretrained).
- **Category Embedding:** Chuyển đổi class ID thành vector nhúng 768 chiều và cộng vào bottleneck của mạng U-Net.
- **Decoder:** Khối 3 cấp `UpBlock` kết hợp `nn.Upsample(scale_factor=4)`.
- **Head:** Lớp `Conv2d(64, 1, 1)` trả về logits nhị phân kích thước `[B, 1, 224, 224]`.
- **Hàm mất mát chính:** Occlusion-Aware Loss (phạt 5x trên vùng bị che khuất) + Soft Dice Loss.

---

## 🗂️ Cấu Trúc Mã Nguồn

```text
.
├── COLAB_A100_GUIDE.md         # 📖 Hướng dẫn chạy trọn gói trên Google Colab A100 & lưu Drive
├── train_d2sa_colab_a100.ipynb # 🚀 Notebook 1-click huấn luyện 6 cấu hình trên Colab
├── README.md                   # Giới thiệu tổng quan dự án
├── requirements.txt            # Thư viện phụ thuộc
├── app.py                      # Giao diện tương tác Gradio Web UI
├── assets/                     # Hình ảnh minh họa & sơ đồ
├── docs/                       # Tài liệu nghiên cứu khoa học cốt lõi
│   ├── PROJECT_UNDERSTANDING.md
│   ├── PROPOSED_METHOD.md
│   ├── DATASET_INSTANCES_DETAILED.md
│   └── TRAINING_SUMMARY.md
└── scripts/                    # Mã nguồn huấn luyện & đánh giá
    ├── model.py                # Kiến trúc Swin-UNet (Cấu hình chính - Row 4)
    ├── dataset_d2sa.py         # Dataset loader cho MVTec D2SA
    ├── train_d2sa.py           # Huấn luyện D2SA (Row 4)
    ├── evaluate_d2sa.py        # Đánh giá D2SA (mIoU, Dice, Precision, Recall, Inv-mIoU)
    ├── verify_d2sa_dataset.py  # Kiểm tra tính toàn vẹn dataset D2SA
    ├── logging_utils.py        # Logging tức thời (flush, fsync, SIGTERM, PosPixels)
    ├── run_ablation_d2sa.py    # Runner tự động chạy và tổng hợp Table V
    ├── run_ablation_d2sa.sh    # Bash script chạy cụm GPU / Colab
    ├── other_config_d2sa/      # 5 cấu hình ablation còn lại cho D2SA (Row 1, 2, 3, 5, 6)
    ├── other_config_cocoa/     # Các cấu hình ablation cho COCOA
    ├── dataset.py              # Dataset loader cho COCOA
    ├── train.py                # Huấn luyện cho COCOA
    ├── evaluate.py             # Đánh giá cho COCOA
    └── train_kins.ipynb        # Huấn luyện trên KINS
```

---

## 📊 Bảng Thực Nghiệm Ablation Study (Table V)

Dự án hỗ trợ 6 cấu hình ablation đối chiếu trên 3 tập dữ liệu (**COCOA**, **KINS**, **D2SA**):

| Row | Cấu hình | Kênh vào | Edge Mask | Category Embedding | Spatial Attention | Hàm mất mát (Loss) | Số tham số |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Baseline (No Edge, No Emb) | 4 | ❌ | ❌ | ❌ | BCE | 34,352,027 |
| **2** | + Edge Mask | 5 | ✅ | ❌ | ❌ | BCE | 34,353,563 |
| **3** | + Occlusion-Aware Loss | 5 | ✅ | ❌ | ❌ | Occ-Aware (5x) + Dice | 34,353,563 |
| **4** | **Main Config (+ Category Embedding)** | 5 | ✅ | ✅ | ❌ | Occ-Aware (5x) + Dice | **34,399,643** |
| **5** | + Spatial Attention | 5 | ✅ | ❌ | ✅ | Occ-Aware (5x) + Dice | 34,353,661 |
| **6** | Full Config (Emb + Spatial) | 5 | ✅ | ✅ | ✅ | Occ-Aware (5x) + Dice | 34,399,741 |