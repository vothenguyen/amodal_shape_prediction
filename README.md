# 👁️ Amodal Shape Prediction (Swin-UNet + SAM 2.1)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue.svg?style=flat-square&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-UI-blue.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)

Báo cáo dự án Nghiên cứu Khoa học / Đồ án chuyên ngành Khoa học Máy tính.
Dự án xây dựng mô hình phân đoạn **Amodal Mask**: dự đoán vùng vật thể đầy đủ ngay cả khi chúng bị lấp hoặc che khuất (occlusion) bởi các vật thể khác. 

Hệ thống được thiết kế theo dạng Pipeline 2 giai đoạn (2-Stage) kết hợp sức mạnh Zero-shot của **Segment Anything Model (SAM 2.1)** và khả năng học biểu diễn toàn cục của **Swin Transformer + U-Net**.

![Demo App](assets/screenshots/demo.png) *(Giao diện web Gradio của hệ thống)*

---

## 🌟 1. Tính năng & Mục tiêu cốt lõi
- **Tương tác trực quan:** Hỗ trợ click chuột (Point Prompt) trực tiếp trên ảnh để chọn vật thể. Hỗ trợ dự đoán cho 90 lớp vật thể chuẩn COCO.
- **Pipeline 2-Stage mạnh mẽ:**
  - *Stage 1 (Phân đoạn cục bộ):* Dùng `SAM 2.1` để trích xuất mặt nạ hiển thị (Visible Mask) từ điểm click.
  - *Stage 2 (Suy luận Amodal):* Dùng `Swin-UNet` phân tích đa kênh để dự đoán toàn bộ hình dáng vật thể.
- **Tính toán hình học:** Tự động đối chiếu Visible Mask và Amodal Mask để tính toán diện tích che khuất và hiển thị trực quan phần bị lấp.

## 🧠 2. Kiến trúc Mô hình (Amodal Swin-UNet)
Mô hình đã được nâng cấp từ phiên bản 4-channel cũ lên cấu trúc 5-channel để tăng cường *Inductive Bias* cho mạng nơ-ron:

- **Encoder:** Sử dụng backbone `timm` (`swin_tiny_patch4_window7_224`, pretrained). Patch Embedding được can thiệp sửa đổi để nhận **Input Tensor [B, 5, 224, 224]** bao gồm:
  1. `Kênh 1-3:` Ảnh RGB (đã chuẩn hóa).
  2. `Kênh 4:` Visible Mask (phần vật thể không bị che khuất).
  3. `Kênh 5:` Edge Mask (ranh giới bị lấp, trích xuất bằng thuật toán hình thái học cv2.dilate/erode).
- **Decoder:** Khối 3 cấp `UpBlock` kết hợp `nn.Upsample(scale_factor=4)`.
- **Head:** Lớp `Conv2d(64, 1, 1)` trả về logits. Kích thước output cuối cùng: `[B, 1, 224, 224]`.

## 🗂️ 3. Dữ liệu & Tiền xử lý
- **Định dạng:** Tương tự COCOA format (`COCO_amodal_train2014.json`). Annotations chứa `regions`, `segmentation`, và `order`.
- **Logic xử lý (Dataset):**
  - Xóa phần bị che khuất dựa vào trường `order` (vật có order thấp hơn sẽ che vật cao hơn) để tạo Visible Mask mô phỏng.
  - Sử dụng `albumentations` để resize đồng bộ (Image + Mask) về `224x224`.

## 📂 4. Cấu trúc mã nguồn
Dự án được tổ chức theo tiêu chuẩn Separation of Concerns:

```text
.
├── app.py                  # Điểm khởi chạy giao diện tương tác (Gradio)
├── assets/                 # Hình ảnh minh họa (figures, screenshots)
├── checkpoints/            # Lưu trữ weights (sam2.1_b.pt, swin_amodal_epoch_30.pth)
├── docs/                   # Tài liệu báo cáo, phân tích chiến lược occlusion
├── notebooks/              # Jupyter notebooks dùng để EDA dữ liệu & training (Colab)
├── outputs/                # Thư mục lưu kết quả tự động (JSON, PNG) từ các script
├── scripts/                # Kịch bản thực thi: train.py, evaluate.py, ablation_study.py...
├── src/                    # Mã nguồn lõi (dataset.py, model.py, advanced_loss.py...)
└── tests/                  # Mã nguồn kiểm thử (Unit test)