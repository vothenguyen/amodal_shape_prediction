# PHƯƠNG PHÁP ĐỀ XUẤT: DỰ ĐOÁN HÌNH DẠNG AMODAL (AMODAL SHAPE PREDICTION)

## 1. TỔNG QUAN PHƯƠNG PHÁP

### 1.1 Bài toán và Động lực

Bài toán dự đoán hình dạng amodal (Amodal Shape Prediction) được định nghĩa như sau:

**Đầu vào:** 
- Ảnh RGB của cảnh chứa các vật thể che khuất lẫn nhau
- Điểm click hoặc vùng quan tâm để chỉ định vật thể cần dự đoán
- Thông tin loại vật thể (category)

**Đầu ra:** 
- Mặt nạ nhị phân (binary mask) biểu diễn hình dạng toàn bộ của vật thể, bao gồm cả phần bị che khuất (occlusion region)

**Ứng dụng thực tiễn:**
- Hiểu rõ cấu trúc vật thể trong cảnh phức tạp
- Tính toán diện tích, kích thước thực tế của vật thể
- Cải thiện độ chính xác của các hệ thống nhận diện đối tượng
- Ứng dụng trong robot vision, autonomous driving, medical imaging

### 1.2 Kiến trúc Pipeline 2-Stage (Two-Stage Pipeline)

Phương pháp đề xuất được thiết kế theo kiến trúc giai đoạn kép:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Phân đoạn cục bộ (Local Segmentation)                 │
│                                                                 │
│ Input: Ảnh RGB + Point Prompt (click chuột)                   │
│ Output: Visible Mask (phần nhìn thấy của vật thể)             │
│                                                                 │
│ Model: Segment Anything Model 2.1 (SAM 2.1) - Zero-shot      │
│        Không cần huấn luyện, dùng promptsinteractively       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Suy luận Amodal (Amodal Inference)                    │
│                                                                 │
│ Input: RGB + Visible Mask + Edge Mask + Category ID            │
│ Output: Amodal Mask (hình dạng toàn bộ)                        │
│                                                                 │
│ Model: Swin Transformer Encoder + U-Net Decoder                │
│        + Spatial Attention Module (Học từ dữ liệu COCO-Amodal)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Post-processing: Tính toán hình học                            │
│                                                                 │
│ - Occlusion Region = Amodal Mask - Visible Mask               │
│ - Diện tích che khuất, tỷ lệ occlusion                         │
│ - Visualize kết quả                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. STAGE 1: PHÂN ĐOẠN CỤC BỘ VỚI SAM 2.1

### 2.1 Tại sao sử dụng SAM 2.1?

SAM 2.1 (Segment Anything Model 2.1) là một mô hình foundation được huấn luyện trên tập dữ liệu khổng lồ với khả năng:

1. **Zero-shot segmentation:** Có thể phân đoạn bất kỳ vật thể nào mà không cần huấn luyện thêm
2. **Prompt flexibility:** Hỗ trợ nhiều dạng prompt (point, box, mask, text)
3. **Tốc độ nhanh:** Inference nhanh cho các ứng dụng real-time
4. **Độ chính xác cao:** Huấn luyện trên 11 triệu ảnh với 1.1 tỷ masks

### 2.2 Quy trình Stage 1

**Input:** Ảnh RGB + Tọa độ điểm click (x, y)

**Bước 1: Chuẩn bị dữ liệu**
- Đọc ảnh gốc từ tệp
- Chuyển đổi từ định dạng BGR (OpenCV) sang RGB
- Chuẩn hóa về khoảng giá trị [0, 255]

**Bước 2: Tạo point prompt**
- Tạo prompt từ tọa độ click: `prompt = {"point_coords": [[x, y]], "point_labels": [1]}`
- Label 1 = điểm thuộc vật thể (positive), 0 = điểm ngoài vật thể (negative)

**Bước 3: Inference với SAM 2.1**
```
visible_mask = sam_model.predict(image, point_prompt)
```
- SAM trả về mask nhị phân (0/1) của vật thể
- Kích thước output bằng kích thước ảnh gốc

**Output:** Visible Mask (phần nhìn thấy)

### 2.3 Ưu điểm của Stage 1

- ✅ Không cần labeled data để huấn luyện Stage 1
- ✅ Khả năng generalize cao (zero-shot)
- ✅ Tương tác trực quan (user-friendly)
- ✅ Độc lập với Stage 2 (có thể thay thế model khác)

---

## 3. STAGE 2: DỰ ĐOÁN AMODAL VỚI SWIN-UNET

Đây là thành phần chính của phương pháp, được huấn luyện trên tập dữ liệu COCO-Amodal.

### 3.1 Xử lý dữ liệu

#### 3.1.1 Chuẩn bị Input 5-Kênh

Từ ảnh gốc, chúng tôi xây dựng một tensor 5-kênh `[B, 5, 224, 224]`:

| Kênh | Tên | Mô tả | Nguồn dữ liệu |
|------|-----|-------|---------------|
| 1-3 | RGB Image | Ảnh màu chuẩn hóa | Ảnh gốc từ dataset |
| 4 | Visible Mask | Phần vật thể nhìn thấy | SAM 2.1 hoặc annotation |
| 5 | Edge Mask | Ranh giới của vùng che | Morphological operations |

**Quá trình tính Edge Mask:**

```python
# Visible mask đã có từ Stage 1
visible_mask = sam_output  # Binary mask [H, W]

# Amodal mask từ annotation
amodal_mask = load_from_json()  # Binary mask [H, W]

# Tính vùng bị che khuất (occlusion)
occlusion_region = amodal_mask - visible_mask  # [0, 1]

# Edge mask: ranh giới của vùng bị che
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
occlusion_dilated = cv2.dilate(occlusion_region, kernel, iterations=2)
edge_mask = occlusion_dilated - occlusion_region  # Ranh giới
```

**Lợi ích của 5-kênh:**
1. **RGB:** Thông tin đặc trưng chính của cảnh
2. **Visible Mask:** Gợi ý vùng vật thể đã biết
3. **Edge Mask:** Hướng dẫn mô hình về ranh giới dự đoán

#### 3.1.2 Thống kê Dataset

**Định nghĩa Instance:** Một instance = một cặp (ảnh RGB, Visible Mask tương ứng)

| Thước tính | Training Set | Validation Set |
|-----------|--------------|----------------|
| **Số instances** | 22,163 mẫu | 12,753 mẫu |
| **Kích thước ảnh** | Đã dạng (resize về 224×224) | Đã dạng (resize về 224×224) |
| **Số classes** | 91 COCO classes | 91 COCO classes |
| **Không che khuất** | 9,379 mẫu (42.3%) | ~4,756 mẫu (37.3%) |
| **Che nhẹ (1-10%)** | 8,348 mẫu (37.7%) | Tương ứng |
| **Che vừa (10-25%)** | 3,023 mẫu (13.6%) | Tương ứng |
| **Che nặng (>25%)** | 1,413 mẫu (6.4%) | Tương ứng |

**Ghi chú quan trọng:**
- Mỗi ảnh có thể chứa nhiều vật thể → mỗi vật thể = 1 instance riêng biệt
- **Visible Mask** được tính tự động từ annotation (không phải từ user input)
- **Amodal Mask** là nhãn (label) mà mô hình cần dự đoán
- **Occlusion Region** = Amodal Mask - Visible Mask (phần bị che khuất)

#### 3.1.3 Dataset Xây dựng

**Nguồn dữ liệu:** COCO-Amodal Dataset

- **Tập training:** ~120K ảnh từ COCO train2014 → **22,163 instances** sau bóc tách
- **Tập validation:** ~20K ảnh từ COCO val2014 → **12,753 instances** sau bóc tách
- **Số lớp vật thể:** 91 loại (theo tiêu chuẩn COCO)

**Bóc tách mẫu:** 

**Vấn đề:** Dataset COCO ban đầu có:
- Training set: ~120K ảnh (COCO train2014)
- Validation set: ~20K ảnh (COCO val2014)

Mỗi ảnh có thể chứa nhiều vật thể (trung bình 1-2 vật thể/ảnh). Nếu coi mỗi ảnh = 1 mẫu training, dataset sẽ nhỏ.

**Giải pháp:** Bóc tách thành instances
```
1 ảnh = N instances (N = số vật thể trong ảnh)

Ví dụ:
- Ảnh 1 có 1 người → 1 instance
- Ảnh 2 có 1 người + 1 chiếc ghế → 2 instances
- Ảnh 3 có 2 người + 2 chiếc bàn → 4 instances

Kết quả:
- Training: ~120K ảnh → 22,163 instances
- Validation: ~20K ảnh → 12,753 instances
```

**Lợi ích:**
- ✅ Tăng kích thước dataset 2-3 lần
- ✅ Mỗi vật thể được xử lý độc lập
- ✅ Cân bằng phân bố các loại vật thể
- ✅ Mỗi instance có nhãn (label) riêng: amodal mask của vật thể đó

**Dự xử lý ảnh:**
- Chuẩn hóa kích thước: 224×224 (tiêu chuẩn cho Swin Transformer)
- Chuẩn hóa pixel: ImageNet normalization (mean, std)
- Data augmentation (khi training):
  - Lật ngang (Horizontal flip): 50%
  - Biến đổi hình học (Shift, Scale, Rotate): 50%
  - Thay đổi độ sáng/tương phản: 20%

#### 3.1.3 Xây dựng Visible và Occlusion Mask

Từ annotation COCO-Amodal:

```python
# Bước 1: Vẽ Amodal Mask (toàn bộ vật thể)
amodal_mask = np.zeros((height, width), dtype=np.uint8)
for polygon in target_region["segmentation"]:
    poly_2d = np.array(polygon).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(amodal_mask, [poly_2d], 1)

# Bước 2: Vẽ Visible Mask (xóa phần bị che)
visible_mask = amodal_mask.copy()
target_order = target_region.get("order", 0)

# Xóa các vật thể phía trước (order nhỏ hơn)
for other_region in ann["regions"]:
    if other_region.get("order", 0) < target_order:
        other_seg = other_region.get("segmentation", [])
        for polygon in other_seg:
            poly_2d = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(visible_mask, [poly_2d], 0)  # Xóa bỏ

# Bước 3: Tính Occlusion Region
occlusion_region = (amodal_mask - visible_mask).astype(np.uint8)
```

### 3.2 Kiến trúc Mô hình Chi tiết

#### 3.2.1 Encoder: Swin Transformer

```
Input: [B, 5, 224, 224]
    ↓
Patch Embedding (Modified):
  - Conv2d: 5 → 96 channels, kernel=4, stride=4
  - Output: [B, 96, 56, 56]
    ↓
Swin Transformer Backbone (4 stages):
  - Stage 1: [B, 96, 56, 56]   → Attention window 7×7
  - Stage 2: [B, 192, 28, 28]  → Merged patches
  - Stage 3: [B, 384, 14, 14]  → Merged patches
  - Stage 4: [B, 768, 7, 7]    → Merged patches (Bottleneck)
    ↓
Output: 4 feature maps với độ phân giải giảm dần
```

**Điểm đặc biệt:**
- Swin Transformer base model: `swin_tiny_patch4_window7_224` từ timm
- Pre-trained trên ImageNet-1K
- **Cải thiện Patch Embedding để xử lý 5 kênh:**
  - Lớp patch embedding gốc chỉ xử lý 3 kênh (RGB)
  - Tạo layer mới với kernel: Conv2d(5, 96, kernel=4, stride=4)
  - Sao chép trọng số pre-trained cho 3 kênh RGB
  - Khởi tạo ngẫu nhiên cho 2 kênh bổ sung (Visible + Edge)

#### 3.2.2 Category Embedding: Nhúng thông tin lớp

```python
# Chuyển đổi class ID thành vector nhúng
category_embedding = nn.Embedding(num_classes=91, embedding_dim=768)

# Trong forward pass:
class_id = 3  # Ví dụ: lớp "Car"
c_emb = category_embedding(class_id)  # [768]
c_emb = c_emb.unsqueeze(-1).unsqueeze(-1)  # [768, 1, 1]

# Thêm vào bottleneck của encoder
x_bottleneck = x_bottleneck + c_emb  # Broadcasting
```

**Lợi ích:**
- Gợi ý cho mô hình về loại vật thể (hình dáng điển hình)
- Giúp các lớp vật thể khác nhau có biểu diễn khác nhau
- Tăng độ chính xác dự đoán cho từng loại

#### 3.2.3 Decoder: U-Net với Skip Connections

```
Encoder bottleneck [B, 768, 7, 7]
    ↓
UpBlock 1:
  - Upsample: [B, 768, 7, 7] → [B, 384, 14, 14]
  - Skip connection từ encoder: [B, 384, 14, 14]
  - Concatenate: [B, 768, 14, 14]
  - DoubleConv: [B, 768, 14, 14] → [B, 384, 14, 14]
    ↓
UpBlock 2:
  - Upsample: [B, 384, 14, 14] → [B, 192, 28, 28]
  - Skip connection: [B, 192, 28, 28]
  - Concatenate: [B, 384, 28, 28]
  - DoubleConv: [B, 384, 28, 28] → [B, 192, 28, 28]
    ↓
UpBlock 3:
  - Upsample: [B, 192, 28, 28] → [B, 96, 56, 56]
  - Skip connection: [B, 96, 56, 56]
  - Concatenate: [B, 192, 56, 56]
  - DoubleConv: [B, 192, 56, 56] → [B, 96, 56, 56]
    ↓
Up-final:
  - Upsample 4x: [B, 96, 56, 56] → [B, 96, 224, 224]
  - Conv: [B, 96, 224, 224] → [B, 64, 224, 224]
  - BatchNorm + ReLU
    ↓
Output Logits:
  - Conv1x1: [B, 64, 224, 224] → [B, 1, 224, 224]
```

**Thiết kế Skip Connections:**
- Kết nối các lớp encoder tương ứng với decoder
- Giữ lại thông tin chi tiết từ các độ phân giải khác nhau
- Giải quyết vanishing gradient problem

#### 3.2.4 Spatial Attention Module

```python
class SpatialAttention(nn.Module):
    """
    Cơ chế chú ý không gian để tập trung vào vùng quan trọng.
    """
    def forward(self, x):
        # x: [B, C, H, W]
        
        # Tính trung bình và max theo chiều kênh
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Nối 2 giá trị
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # Tạo bản đồ trọng số
        scale = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        
        # Nhân với input
        return x * scale
```

**Mục đích:**
- Tập trung vào vùng vật thể (foreground)
- Giảm thiểu ảnh hưởng của background
- Học tự động trọng số không gian từ dữ liệu

### 3.3 Hàm Loss Function

#### 3.3.1 Occlusion-Aware Loss

Vấn đề: Vùng bị che khuất chiếm **ít pixel** hơn vùng nhìn thấy (class imbalance)

**Giải pháp:** Sử dụng weighted loss function

```python
class OcclusionAwareLoss(nn.Module):
    def forward(self, pred, target, occluded_region):
        # 1. Tính BCE loss cho từng pixel
        bce_loss = self.bce(pred, target)  # [B, 1, H, W]
        
        # 2. Tạo ma trận trọng số
        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = 5.0  # 5x weight cho occlusion
        
        # 3. Áp dụng trọng số
        weighted_bce = (bce_loss * weight_matrix).mean()
        
        # 4. Thêm Dice loss để cân bằng
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        # 5. Kết hợp
        total_loss = weighted_bce + dice_loss.mean()
        return total_loss
```

**Công thức chi tiết:**

$$\text{Loss} = \text{BCE}_{\text{weighted}} + \text{Dice}_{\text{loss}}$$

Trong đó:

$$\text{BCE}_{\text{weighted}} = -\frac{1}{HW} \sum_{i,j} w_{ij} [y_{ij} \log(p_{ij}) + (1-y_{ij}) \log(1-p_{ij})]$$

- $w_{ij} = 1$ nếu pixel $(i,j)$ ở vùng không bị che
- $w_{ij} = 5$ nếu pixel $(i,j)$ ở vùng bị che

$$\text{Dice}_{\text{loss}} = 1 - \frac{2|X \cap Y|}{|X| + |Y|}$$

**Lợi ích:**
- ✅ Tự động cân bằng class imbalance
- ✅ Tập trung vào vùng khó dự đoán (occlusion)
- ✅ Kết hợp 2 metrics bổ sung cho nhau

---

## 4. QUY TRÌNH HUẤN LUYỆN

### 4.1 Cấu hình Huấn luyện

| Tham số | Giá trị | Ghi chú |
|---------|--------|---------|
| Model | Swin-UNet | swin_tiny_patch4_window7_224 + U-Net decoder |
| Batch Size | 4 | Actual batch |
| Gradient Accumulation | 4 steps | Effective batch = 16 |
| Epochs | 30 | Có thể resume từ checkpoint |
| Learning Rate | 1e-4 | AdamW optimizer |
| LR Schedule | Cosine Annealing | $T_{max} = 30$ epochs |
| Loss Function | OcclusionAwareLoss | Weighted BCE + Dice |
| Data Augmentation | Albumentations | Flip, Shift, Rotate, Brightness/Contrast |

### 4.2 Gradient Accumulation

**Tại sao sử dụng?**
- GPU memory giới hạn → batch size nhỏ (4)
- Gradient accumulation để mô phỏng batch size lớn (16)

**Cách hoạt động:**

```python
for epoch in range(EPOCHS):
    optimizer.zero_grad()  # Xóa gradient cũ
    
    for i, (input, target, occluded) in enumerate(train_loader):
        # Forward pass
        output = model(input, class_ids)
        loss = criterion(output, target, occluded)
        
        # Backward pass (chia loss cho ACCUMULATION_STEPS)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()  # Tích lũy gradient
        
        # Update weights mỗi ACCUMULATION_STEPS hoặc cuối batch
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
```

### 4.3 Learning Rate Schedule

Sử dụng **Cosine Annealing** để giảm learning rate theo thời gian:

$$LR(t) = LR_{min} + \frac{1 + \cos(\pi t / T_{max})}{2} (LR_{init} - LR_{min})$$

```
LR
 ↑
 │     
1e-4 ├─────┐
     │      \
     │       \
     │        └────
 0   └────────────→ Epoch
     0    15    30
```

**Lợi ích:**
- ✅ Giảm từ từ learning rate → tối ưu chính xác hơn
- ✅ Tránh overfitting ở các epoch sau
- ✅ Hội tụ nhanh hơn so với fixed LR

### 4.4 Data Augmentation Strategy

**Mục đích:** Tăng độ đa dạng dữ liệu, giảm overfitting

| Augmentation | Xác suất | Tham số |
|--------------|----------|---------|
| HorizontalFlip | 50% | - |
| ShiftScaleRotate | 50% | shift=0.05, scale=0.1, rotate=15° |
| RandomBrightnessContrast | 20% | brightness=0.1, contrast=0.1 |
| Resize | 100% | 224×224 (đồng bộ ảnh + mask) |

**Lợi ích:**
- ✅ Mô hình học từ nhiều biến thể của dữ liệu
- ✅ Tăng khả năng generalize
- ✅ Giảm overfitting trên training set

### 4.5 Checkpoint Management

Sau mỗi epoch:

```python
save_path = f"checkpoints/swin_amodal_epoch_{epoch+1}.pth"
torch.save(model.state_dict(), save_path)
```

**Lợi ích:**
- ✅ Có thể resume training nếu ngắt giữa chừng
- ✅ Lưu lại các model tốt nhất
- ✅ Dễ dàng so sánh hiệu suất giữa các epoch

---

## 5. METRICS ĐÁNH GIÁ

Để đánh giá chất lượng dự đoán, chúng tôi sử dụng các metric tiêu chuẩn:

### 5.1 Intersection over Union (IoU)

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

Trong đó:
- A = predicted mask
- B = ground truth amodal mask

**Giải thích:** Tỉ lệ vùng giao giữa prediction và ground truth trên tổng hợp.

**Range:** 0 → 1 (càng cao càng tốt)

### 5.2 Dice Coefficient (F1-Score)

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

**So sánh với IoU:**
- Dice nhạy cảm hơn với những sai lầm nhỏ
- IoU khắt khe hơn

### 5.3 Invisible IoU (Occlusion-Specific Metric)

$$\text{Invisible IoU} = \frac{|A_{inv} \cap B_{inv}|}{|A_{inv} \cup B_{inv}|}$$

Trong đó:
- $A_{inv}$ = predicted mask ở vùng bị che khuất
- $B_{inv}$ = ground truth amodal mask ở vùng bị che khuất

**Mục đích:** 
- ✅ Đánh giá khả năng dự đoán phần bị che
- ✅ Chỉ ra vùng mô hình gặp khó khăn
- ✅ Hợp lý hơn cho bài toán amodal

### 5.4 Tính toán các Metrics

```python
def calculate_metrics(pred_logits, target, visible, threshold=0.5):
    # Chuyển logit thành binary mask
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    
    # IoU toàn bộ
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice coefficient
    dice = (2.0 * intersection + 1e-6) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
    )
    
    # Invisible IoU (chỉ vùng bị che)
    invisible_target = torch.clamp(target - visible, min=0.0)
    inv_intersection = (pred * invisible_target).sum(dim=(2, 3))
    inv_union = (pred + invisible_target - inv_intersection)
    invisible_iou = (inv_intersection + 1e-6) / (inv_union + 1e-6)
    
    return iou, dice, invisible_iou
```

**Kết quả cuối cùng:**
```
mIoU (mean IoU):           x.xx
mDice (mean Dice):         x.xx
mInvisible IoU (mean):     x.xx   ← Metric quan trọng nhất
```

---

## 6. QUÁN TRÌNH HỒI QUI (INFERENCE)

### 6.1 Inference Pipeline

```
Input:
  - Ảnh RGB [H, W, 3]
  - Class ID (category)
  - (Option) Point click cho SAM
    ↓
Stage 1: SAM 2.1 Segmentation
  - Tạo point prompt từ click
  - Inference: visible_mask = sam(image, prompt)
  - Output: Visible mask [H, W]
    ↓
Pre-processing:
  - Resize: [H, W] → [224, 224]
  - Tính edge mask từ occlusion region
  - Stack 5 channels: [5, 224, 224]
  - Normalize: ImageNet normalization
    ↓
Stage 2: Amodal Swin-UNet
  - model.eval()  # Bật dropout off, batch norm frozen
  - with torch.no_grad():
    - logits = model(input_tensor, class_ids)  # [1, 1, 224, 224]
  - Output: Logits mask
    ↓
Post-processing:
  - Sigmoid: probs = sigmoid(logits)  # [0, 1]
  - Threshold: mask = probs > 0.5  # Binary mask
  - Resize: [224, 224] → [H, W]  # Kích thước gốc
  - Compute occlusion region: occlusion = amodal - visible
    ↓
Output:
  - Amodal mask [H, W]
  - Occlusion region [H, W]
  - Visualization + Metrics
```

### 6.2 Batch Processing

Có thể xử lý nhiều ảnh cùng lúc:

```python
# Input: Batch của 8 ảnh
batch_images = torch.stack([img1, img2, ..., img8])  # [8, 5, 224, 224]
batch_class_ids = torch.tensor([3, 5, 1, ..., 2])    # [8]

# Inference
with torch.no_grad():
    logits = model(batch_images, batch_class_ids)  # [8, 1, 224, 224]

# Post-processing cho cả batch
pred_masks = (torch.sigmoid(logits) > 0.5).float()  # [8, 1, 224, 224]
```

**Tốc độ:**
- Giây/ảnh trên GPU: ~50-100ms
- Giây/ảnh trên CPU: ~500-1000ms

---

## 7. ĐÓNG GÓP CHÍNH CỦA PHƯƠNG PHÁP

### 7.1 Các tính năng sáng tạo

1. **Pipeline 2-Stage tối ưu:**
   - Stage 1 zero-shot + Stage 2 learned → kết hợp điểm mạnh của cả hai
   - Không phụ thuộc vào annotation chính xác từ người dùng

2. **Input 5-Channel thiết kế:**
   - RGB + Visible + Edge → gợi ý đủ thông tin cho mô hình
   - Edge mask giúp mô hình hiểu ranh giới dự đoán

3. **Category Embedding:**
   - Tận dụng thông tin loại vật thể
   - Giúp mô hình học các biểu diễn riêng biệt cho từng lớp

4. **Occlusion-Aware Loss:**
   - Giải quyết class imbalance tự nhiên
   - Tập trung vào vùng khó dự đoán

5. **Spatial Attention:**
   - Tập trung vào vùng vật thể quan trọng
   - Giảm ảnh hưởng của background

### 7.2 Khả năng generalize

- ✅ Hoạt động với 91 lớp vật thể khác nhau
- ✅ Có thể mở rộng sang các dataset khác (Pascal VOC, etc.)
- ✅ Zero-shot SAM stage → không cần huấn luyện lại khi thay đổi dữ liệu

---

## 8. GIỚI HẠN VÀ HƯỚNG PHÁT TRIỂN

### 8.1 Giới hạn hiện tại

1. **Phụ thuộc vào SAM quality:** Nếu Stage 1 sai → Stage 2 khó bù được
2. **Chậm với ảnh độ phân giải cao:** Chỉ xử lý 224×224, cần upsampling
3. **Memory intensive:** Batch size nhỏ (4) do GPU memory
4. **Cần training data:** Yêu cầu COCO-Amodal annotations (110K ảnh)

### 8.2 Hướng phát triển tương lai

1. **Multi-scale processing:** Xử lý đa tỷ lệ để cải thiện chi tiết
2. **Refinement stage:** Thêm stage tinh chỉnh sau Stage 2
3. **Lightweight model:** Chuyển sang mobile-friendly architecture
4. **Few-shot learning:** Học từ ít annotations hơn
5. **End-to-end training:** Tối ưu cả SAM + Swin-UNet chung lúc

---

## 9. TÓM TẮT PHƯƠNG PHÁP

| Khía cạnh | Chi tiết |
|----------|---------|
| **Kiến trúc** | 2-Stage: SAM 2.1 + Swin-UNet |
| **Input** | RGB (3ch) + Visible mask (1ch) + Edge mask (1ch) |
| **Output** | Amodal mask (binary) |
| **Encoder** | Swin Transformer (pre-trained on ImageNet) |
| **Decoder** | 3-layer U-Net + Skip connections |
| **Attention** | Spatial Attention module |
| **Loss** | Weighted BCE + Dice (5x weight cho occlusion) |
| **Training** | 30 epochs, batch=4, gradient accumulation=4 |
| **Learning Rate** | 1e-4, Cosine Annealing schedule |
| **Metrics** | IoU, Dice, Invisible IoU |
| **Data Augmentation** | Flip, Shift, Rotate, Brightness/Contrast |

---

**Kết luận:** Phương pháp đề xuất kết hợp hiệu quả giữa zero-shot SAM 2.1 và học từ dữ liệu với Swin-UNet, cung cấp một giải pháp toàn diện để dự đoán hình dạng amodal của vật thể trong cảnh phức tạp với occlusion.
