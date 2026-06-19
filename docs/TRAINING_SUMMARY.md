# PHẦN 3: XỬ LÝ DỮ LIỆU & HUẤN LUYỆN - TÓM TẮT NGẮN

## Ôn tập: Instance là gì?

```
┌──────────────────────────────────────────────┐
│ MỘT INSTANCE = (ảnh RGB, Visible Mask)      │
│                                              │
│ Ảnh RGB: 3 kênh màu gốc                     │
│ Visible Mask: phần vật thể nhìn thấy         │
│                                              │
│ Input model: [RGB + Visible + Edge] = 5ch   │
│ Output label: Amodal Mask                    │
└──────────────────────────────────────────────┘
```

---

## Thống kê Dataset

### Kích thước Dataset

| | Training | Validation |
|---|----------|-----------|
| Ảnh gốc | ~120K ảnh | ~20K ảnh |
| **Instances** | **22,163** | **12,753** |

### Phân bố Occlusion

**Training Set (22,163 instances):**

```
Không che khuất (0-1%):  ████████░ 42.3%  →  9,379 mẫu (Dễ)
Che nhẹ (1-10%):         █████████ 37.7%  →  8,348 mẫu
Che vừa (10-25%):        ███░░░░░░ 13.6%  →  3,023 mẫu (Khó)
Che nặng (>25%):         ██░░░░░░░ 6.4%   →  1,413 mẫu (Rất khó)
```

**Validation Set:** Phân bố tương tự

---

## Cấu trúc 5-kênh Input

```
[B, 5, 224, 224]
  ├─ Kênh 0-2: RGB Image (ảnh gốc) [0, 1] normalized
  ├─ Kênh 3: Visible Mask (vật thể nhìn thấy) {0, 1}
  └─ Kênh 4: Edge Mask (ranh giới) [0, 1] soft
```

**Lý do 5 kênh:**
- RGB: Thông tin cơ bản (màu sắc, kết cấu)
- Visible: Gợi ý vùng vật thể
- Edge: Gợi ý ranh giới dự đoán

---

## Quá trình tạo một Instance từ Ảnh

### Bước 1: Tải ảnh gốc

```python
image = cv2.imread("train2014/img.jpg")  # [H, W, 3]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### Bước 2: Vẽ Amodal Mask (toàn bộ vật thể)

```python
amodal_mask = np.zeros((height, width), dtype=np.uint8)
for polygon in target_region["segmentation"]:
    cv2.fillPoly(amodal_mask, [polygon], 1)  # Tô đầy
# Result: [H, W] binary mask
```

### Bước 3: Vẽ Visible Mask (xóa phần bị che)

```python
visible_mask = amodal_mask.copy()
for other_region in ann["regions"]:
    if other_region["order"] < target_order:  # Vật phía trước
        cv2.fillPoly(visible_mask, [polygon], 0)  # Xóa bỏ
# Result: [H, W] binary mask (phần nhìn thấy)
```

### Bước 4: Tính Occlusion Region

```python
occlusion_region = amodal_mask - visible_mask
# Phần bị che = amodal - visible
```

### Bước 5: Tạo Edge Mask

```python
kernel = np.ones((5, 5), np.uint8)
edge_mask = cv2.dilate(visible_mask, kernel) - visible_mask
# Ranh giới của vật thể nhìn thấy
```

### Bước 6: Data Augmentation

```python
augmented = transform(
    image=image,
    masks=[amodal_mask, visible_mask],
    transforms=[Resize(224, 224), HorizontalFlip(0.5), ...]
)
```

### Bước 7: Kết hợp thành 5-kênh

```python
input_tensor = torch.cat([
    image_tensor,              # [3, 224, 224] RGB
    visible_mask.unsqueeze(0), # [1, 224, 224] Visible
    edge_mask.unsqueeze(0)     # [1, 224, 224] Edge
], dim=0)
# Result: [5, 224, 224]
```

### Bước 8: Tạo Labels

```python
return {
    'input': input_tensor,      # [5, 224, 224]
    'amodal': amodal_tensor,    # [224, 224] - Label
    'occlusion': occlusion_region,  # [224, 224]
    'class_id': category_id     # scalar
}
```

---

## Huấn Luyện Mô Hình

### Cấu hình Huấn Luyện

| Tham số | Giá trị |
|---------|--------|
| Model | Swin-UNet |
| Batch Size | 4 |
| Gradient Accumulation | 4 steps → Effective batch = 16 |
| Epochs | 30 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
| Loss Function | Weighted BCE + Dice |

### Quá trình Training 1 Epoch

```python
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets, occluded) in enumerate(train_loader):
        
        # Forward pass
        outputs = model(inputs, class_ids)  # [B, 1, 224, 224]
        
        # Loss calculation
        loss = criterion(outputs, targets, occluded)
        
        # Backward pass (gradient accumulation)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Update learning rate
    scheduler.step()
```

### Loss Function: Occlusion-Aware

```
Loss = Weighted BCE + Dice

Weighted BCE:
  - weight = 1.0 cho phần không che
  - weight = 5.0 cho phần bị che ← Tập trung hơn!

Dice Loss:
  - Cân bằng class imbalance
  - F1-score cho segmentation
```

**Tại sao 5x weight cho occlusion?**
```
Vấn đề: 95% pixel không che, 5% bị che
→ Class imbalance cực đoan

Giải pháp: Tăng weight cho phần bị che
→ Mô hình tập trung vào vùng khó dự đoán
→ Improve Invisible IoU (metric quan trọng)
```

### Learning Rate Schedule

```
LR(t) = LR_init × (1 + cos(π × t / T_max)) / 2

Điều chỉnh LR từ 1e-4 → 0 trong 30 epochs
```

### Data Augmentation

```
- Resize: 224×224 (chuẩn hóa)
- HorizontalFlip: 50% (lật ngang)
- ShiftScaleRotate: 50% (biến đổi hình học)
  - Shift: ±5%
  - Scale: ±10%
  - Rotate: ±15°
- Brightness/Contrast: 20% (thay đổi độ sáng)
```

---

## Đánh Giá Mô Hình

### Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **IoU** | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | Chất lượng mask toàn bộ |
| **Dice** | $\frac{2\|A \cap B\|}{\|A\|+\|B\|}$ | F1-score segmentation |
| **Invisible IoU** | IoU tính trên phần bị che | Chất lượng predicting occlusion |

### Kết quả trên Validation Set (Epoch 30)

```
Overall mIoU:        0.8409  (84.09%)
Overall Dice:        0.8984  (89.84%)
Invisible mIoU:      0.5510  (55.10%) ← Metric chính!
```

**Giải thích:**
- mIoU cao: Mô hình tốt trong dự đoán toàn bộ mask
- Invisible IoU thấp hơn: Vùng occlusion khó dự đoán (bình thường)

---

## Inference (Dự đoán trên ảnh mới)

### Quy trình

```
Input: Ảnh RGB + Point click + Class ID
    ↓
Stage 1: SAM 2.1
    ↓
Output: Visible Mask
    ↓
Pre-processing:
  - Resize: [H, W] → [224, 224]
  - Tính edge mask
  - Stack 5 channels
    ↓
Stage 2: Swin-UNet
    ↓
Output: Logits [1, 1, 224, 224]
    ↓
Post-processing:
  - Sigmoid: logits → [0, 1]
  - Threshold > 0.5: → binary mask
  - Resize: [224, 224] → [H, W]
    ↓
Compute Occlusion Region:
  occlusion = amodal - visible
    ↓
Output: Amodal Mask + Occlusion Region
```

### Tốc độ

- GPU (RTX 3090): ~50-100ms/ảnh
- CPU (i7-12700): ~500-1000ms/ảnh

---

## Workflow Toàn Bộ

```
                ┌─────────────────┐
                │ COCO-Amodal     │
                │ Annotations     │
                └────────┬────────┘
                         │
                         ↓
        ┌────────────────────────────────┐
        │ Dataset Loader (Bóc tách)     │
        │ - Tải ảnh gốc                  │
        │ - Vẽ amodal & visible masks    │
        │ - Tạo edge mask                │
        │ - Data augmentation            │
        │ → Instance [5, 224, 224]      │
        └────────────┬───────────────────┘
                     │
                     ↓
    ┌──────────────────────────────┐
    │ Batch Training (22,163 instances)
    └──────────────┬───────────────┘
                   │
                   ↓
    ┌──────────────────────────────┐
    │ Swin-UNet Model              │
    │ - Encoder: Swin Transformer  │
    │ - Decoder: U-Net + Skip conn │
    │ - Head: Conv1x1 → logits     │
    └──────────────┬───────────────┘
                   │
                   ↓
    ┌──────────────────────────────┐
    │ Loss Calculation             │
    │ - Weighted BCE (5x occlusion)│
    │ - Dice loss (balance)        │
    └──────────────┬───────────────┘
                   │
                   ↓
    ┌──────────────────────────────┐
    │ Backpropagation + Optimization
    │ - AdamW optimizer            │
    │ - Cosine annealing LR        │
    │ - Gradient accumulation      │
    └──────────────┬───────────────┘
                   │
                   ↓
    ┌──────────────────────────────┐
    │ Validation (12,753 instances)│
    │ - Compute metrics: IoU, Dice │
    │ - Invisible IoU              │
    └──────────────┬───────────────┘
                   │
                   ↓
    ┌──────────────────────────────┐
    │ Checkpoint Saved             │
    │ swin_amodal_epoch_X.pth      │
    └──────────────────────────────┘
```

---

## Tóm Tắt Chính

### Dataset
- **22,163 instances training** (từ ~120K ảnh)
- **12,753 instances validation** (từ ~20K ảnh)
- **91 COCO classes**
- **Phân bố occlusion:** 42% không che, 58% có che

### Input
- **5-kênh:** RGB + Visible Mask + Edge Mask
- **Kích thước:** 224×224 (chuẩn hóa)
- **Normalize:** ImageNet standard

### Model
- **Encoder:** Swin Transformer (pre-trained)
- **Decoder:** 3-layer U-Net + Skip connections
- **Attention:** Spatial attention module
- **Output:** Amodal mask [1, 224, 224]

### Training
- **Loss:** Weighted BCE (5x occlusion) + Dice
- **Optimizer:** AdamW, LR=1e-4, Cosine annealing
- **Batch:** 4 (gradient accumulation ×4 = 16)
- **Epochs:** 30

### Metrics
- **mIoU:** 0.8409 (84.09%)
- **Dice:** 0.8984 (89.84%)
- **Invisible IoU:** 0.5510 (55.10%) ← Quan trọng!