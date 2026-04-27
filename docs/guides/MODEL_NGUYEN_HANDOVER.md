# 🚀 Bàn giao Mô hình Amodal Nguyen (Expanding From Active Boundary)

Tài liệu này cung cấp thông tin chi tiết về kiến trúc, input/output và cách khởi tạo mô hình **AmodalPipelineNguyen** nhằm mục đích bàn giao cho bộ phận/cá nhân phụ trách bước **Evaluation (Đánh giá)**.

---

## 1. Tổng quan Kiến trúc (Architecture)

Mô hình được xây dựng dựa trên ý tưởng bài báo *"Expanding From Active Boundary with Compatible Prior"*, bao gồm **4 Module chính** chạy nối tiếp nhau:

1. **Model Base (Swin U-Net)**: 
   - Sử dụng backbone `swin_tiny_patch4_window7_224` (thông qua `timm`) để trích xuất đặc trưng chung (Feature $F$).
   - Có một Decoder nhánh phụ dể dự đoán mask của phần bị che khuất / phần nhìn thấy được (Visible Mask $M^v$).
2. **Non-local Active Boundary Estimator**: 
   - Nhận đầu vào là đặc trưng $F$ và mask $M^v$.
   - Dự đoán ranh giới mở rộng của vật thể (Boundary Mask $M^b$).
3. **Boundary-Aware Shape Prior Bank**: 
   - Đóng vai trò như một cơ sở dữ liệu (codebook) chứa các hình dạng tiên nghiệm (shape prior) của các lớp vật thể.
   - Sử dụng Attention Query (từ $M^v$ và $M^b$) để truy xuất ra hình dạng tổng quát (Shape Prior $M^p$) phù hợp nhất.
4. **Context-Aware Amodal Mask Refiner**: 
   - Nhận đầu vào gồm ngữ cảnh không gian (từ $F$), ranh giới $M^b$, và hình dạng tiên nghiệm $M^p$.
   - Tổng hợp và dự đoán ra kết quả cuối cùng: **Amodal Mask ($M^a$)**.

---

## 2. Đặc tả Input và Output

Khi thực hiện inference / evaluation, mô hình cần nhận tuple input cụ thể và sẽ trả ra 4 output maps.

### 2.1. Input (Đầu vào)
Hàm `forward` của `AmodalPipelineNguyen` nhận 2 tham số chính:
1. `image_x` *(Tensor)*: 
   - **Kích thước:** `[Batch_Size, 3, 224, 224]`
   - **Định dạng:** Ảnh màu RGB, các giá trị pixel đã chuẩn hóa về dải `[0.0, 1.0]`.
2. `category_ids` *(Tensor, Optional)*: 
   - **Kích thước:** `[Batch_Size]`
   - **Định dạng:** Kiểu `torch.long`, chứa ID của class (VD: 1 cho person, 2 cho car, v.v., tối đa 91 classes theo COCO).

### 2.2. Output (Đầu ra)
Mô hình trả ra 4 Tensors chưa qua hàm kích hoạt không gian (chưa qua threshold), đều có kích thước `[Batch_Size, 1, 224, 224]`. Khi evaluate cần áp dụng `torch.sigmoid()` và threshold (thường là `> 0.5`) để lấy nhị phân (binary mask).

1. `Ma`: **Amodal Mask** (Dự đoán mask hoàn chỉnh cuối cùng - *Quan trọng nhất cho Evaluate*).
2. `Mv`: **Visible Mask** (Mask phần nhìn thấy được, không bị che khuất).
3. `Mb`: **Boundary Mask** (Mask viền ranh giới vật thể).
4. `Mp`: **Shape Prior Mask** (Mask đặc trưng dự đoán từ Codebook).

> **Lưu ý quan trọng cho bước Evaluate:** Metric (mIoU) chủ yếu sẽ được đo lường trên `Ma` so với Ground Truth Amodal Mask.

---

## 3. Cài đặt Môi trường (Environment Setup)

Người làm evaluation cần chuẩn bị môi trường chạy cài đặt các thư viện sau:

```bash
pip install torch torchvision timm albumentations opencv-python pycocotools matplotlib tqdm
```

**Môi trường phần cứng:** Nên sử dụng máy tính có GPU (CUDA) vì backbone Swin Transformer tốn khá nhiều tài nguyên để tính toán inference.

---

## 4. Hướng dẫn Load Model để Evaluate

File chính liên quan: `src/model_Nguyen.py`

Người tiếp nhận thực hiện evaluation có thể dùng đoạn mã mẫu sau để tải checkpoint và chạy inference:

```python
import torch
from src.model_Nguyen import AmodalPipelineNguyen

# 1. Khởi tạo mô hình
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AmodalPipelineNguyen(num_classes=91)

# 2. Tải Checkpoint đã train
checkpoint_path = 'checkpoints/nguyen_model_epoch_2.pth' # Hoặc đường dẫn từ Google Drive
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# 3. Chuẩn bị dữ liệu và Inference
# Giả sử bạn có 1 batch ảnh: images_tensor (B, 3, 224, 224) 
# và category_ids (B)
with torch.no_grad():
    # Model trả về 4 masks
    Ma_logits, Mv_logits, Mb_logits, Mp_logits = model(images_tensor, category_ids)
    
    # 4. Áp dụng sigmoid để nhận xác suất (0.0 đến 1.0)
    amodal_probs = torch.sigmoid(Ma_logits)
    
    # 5. Phân ngưỡng để lấy Binary Mask
    amodal_predictions = (amodal_probs > 0.5).float()
    
# Từ amodal_predictions này, bạn có thể tính toán các metric như mIoU, Dice Score...
```

---

## 5. Các File / Module liên quan cần xem thêm

- `src/model_Nguyen.py`: Mã nguồn chính của cấu trúc mô hình.
- `src/dataset_nguyen.py`: Cách DataLoader xử lý ảnh đầu vào và resize về dạng `224x224`. Có thể tham khảo phần transform để đảm bảo hàm Evaluate load data tương tự.
- `src/loss_nguyen.py`: Cách hệ thống loss tính mIoU/Dice ở bước training, có thể tái sử dụng logic tính này cho report evaluation.
- `src/evaluate.py`: Chứa các hàm base có sẵn của project để chạy tính toán overall mIoU, unseen mIoU. Có thể sửa và trỏ qua mô hình Nguyễn.

Chúc quá trình Evaluation đạt kết quả tốt nhất! 🎯