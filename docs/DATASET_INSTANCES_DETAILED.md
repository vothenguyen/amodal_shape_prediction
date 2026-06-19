# DATASET & INSTANCES - CHI TIẾT COCO-AMODAL

## 1. ĐỊNH NGHĨA INSTANCE

### 1.1 Instance là gì?

Trong bối cảnh của dự án này:

**MỘT INSTANCE = một cặp (ảnh RGB, Visible Mask tương ứng)**

```
┌──────────────────────────────────────────────────────────────┐
│ Instance (1 mẫu huấn luyện)                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                          Output (Label):            │
│  ┌──────────────────────┐       ┌──────────────────────┐   │
│  │                      │       │                      │   │
│  │   ảnh RGB            │       │  Amodal Mask         │   │
│  │   5 kênh:            │   →   │  (ground truth)       │   │
│  │  - RGB (3)           │       │                      │   │
│  │  - Visible (1)       │       │  + Occlusion region  │   │
│  │  - Edge (1)          │       │  = Amodal - Visible  │   │
│  │                      │       │                      │   │
│  └──────────────────────┘       └──────────────────────┘   │
│     [5, 224, 224]                  [1, 224, 224]           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Điểm quan trọng

1. **Mỗi vật thể = một instance riêng biệt**
   ```
   Ảnh có 3 vật thể → 3 instances
   ↓
   Mỗi instance được xử lý độc lập trong training
   ```

2. **Visible Mask đã được tính toán** (không phải từ user)
   - Trong dataset, Visible Mask được tính tự động từ annotation
   - Dựa trên `order` của vật thể (vật phía trước che vật phía sau)
   
3. **Amodal Mask là nhãn (label)**
   - Đó chính là output mà mô hình cần dự đoán
   - Bao gồm cả phần bị che khuất

---

## 2. THỐNG KÊ DATASET

### 2.1 Bảng tổng hợp

| Thước tính | Training Set | Validation Set |
|-----------|--------------|----------------|
| **Số instances** | 22,163 mẫu | 12,753 mẫu |
| **Kích thước ảnh** | Đã dạng (resize về 224×224) | Đã dạng (resize về 224×224) |
| **Số classes** | 91 COCO classes | 91 COCO classes |
| **Không che khuất** | 9,379 mẫu (42.3%) | ~4,756 mẫu (37.3%) |
| **Che nhẹ (1-10%)** | 8,348 mẫu (37.7%) | Tương ứng |
| **Che vừa (10-25%)** | 3,023 mẫu (13.6%) | Tương ứng |
| **Che nặng (>25%)** | 1,413 mẫu (6.4%) | Tương ứng |

### 2.2 Phân bố Occlusion chi tiết

#### Training Set

```
Không che khuất:  ████████░ 42.3%  (9,379 mẫu)
Che nhẹ (1-10%):  █████████ 37.7%  (8,348 mẫu)
Che vừa (10-25%): ███░░░░░░ 13.6%  (3,023 mẫu)
Che nặng (>25%):  ██░░░░░░░ 6.4%   (1,413 mẫu)
                  ─────────────────────────────
                  Tổng: 22,163 mẫu
```

#### Validation Set

```
Không che khuất:  ███████░░ 37.3%  (4,756 mẫu)
Che nhẹ (1-10%):  Tương ứng
Che vừa (10-25%): Tương ứng
Che nặng (>25%):  Tương ứng
                  ─────────────────────────────
                  Tổng: 12,753 mẫu
```

### 2.3 Tổng hợp

- **Tổng instances:** 22,163 (training) + 12,753 (validation) = **34,916 mẫu**
- **Tỷ lệ training/validation:** 63.4% / 36.6%
- **Các class:** Tất cả 91 loại COCO đều có mặt trong cả 2 tập

---

## 3. QUẦN TRÌNH TẠO INSTANCE

### 3.1 Từ Dataset COCO-Amodal

**Nguồn:** COCO train2014 và COCO val2014 với amodal annotations

**Cấu trúc annotation JSON:**

```json
{
  "images": [
    {"id": 1, "file_name": "train2014/img.jpg", "height": 480, "width": 640},
    ...
  ],
  "annotations": [
    {
      "id": 1001,
      "image_id": 1,
      "category_id": 3,
      "regions": [
        {
          "segmentation": [[x1,y1, x2,y2, ...]],  // Amodal shape
          "order": 0,
          "category_id": 3
        },
        {
          "segmentation": [[...]],
          "order": 1,
          "category_id": 5
        }
      ]
    }
  ]
}
```

### 3.2 Quy trình tạo một instance

```python
# BƯỚC 1: Tải ảnh gốc
image = cv2.imread("train2014/img.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

# BƯỚC 2: Vẽ Amodal Mask (toàn bộ vật thể)
amodal_mask = np.zeros((height, width), dtype=np.uint8)
for polygon in target_region["segmentation"]:
    poly_2d = np.array(polygon).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(amodal_mask, [poly_2d], 1)
# Result: Binary mask [H, W] với 1 = vật thể, 0 = background

# BƯỚC 3: Vẽ Visible Mask (xóa phần bị che)
visible_mask = amodal_mask.copy()
target_order = target_region.get("order", 0)

for other_region in ann["regions"]:
    other_order = other_region.get("order", 0)
    if other_order < target_order:  # Vật thể khác phía trước
        for polygon in other_region["segmentation"]:
            poly_2d = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(visible_mask, [poly_2d], 0)  # Xóa
# Result: Binary mask [H, W] với phần bị che = 0

# BƯỚC 4: Tính Occlusion Region
occlusion_region = amodal_mask - visible_mask
# Result: Phần bị che khuất

# BƯỚC 5: Tạo Edge Mask (ranh giới)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(visible_mask * 255, kernel, iterations=1)
erosion = cv2.erode(visible_mask * 255, kernel, iterations=1)
edge_mask = (dilation - erosion) / 255.0
# Result: Viền của visible mask

# BƯỚC 6: Data Augmentation (Resize + Transform)
augmented = albumentations.apply(
    image=image,
    masks=[amodal_mask, visible_mask],
    transforms=[Resize(224, 224), HorizontalFlip(p=0.5), ...]
)

# BƯỚC 7: Kết hợp thành 5 kênh
input_tensor = torch.cat([
    image_tensor,              # Kênh 0-2: RGB [3, 224, 224]
    visible_mask.unsqueeze(0), # Kênh 3: Visible [1, 224, 224]
    edge_mask.unsqueeze(0)     # Kênh 4: Edge [1, 224, 224]
], dim=0)
# Result: [5, 224, 224]

# BƯỚC 8: Tạo instance
instance = {
    'input': input_tensor,           # [5, 224, 224]
    'amodal_label': amodal_tensor,   # [224, 224]
    'occlusion_region': occlusion_region,  # [224, 224]
    'class_id': category_id           # scalar
}
```

---

## 4. CẤU TRÚC CỦA MỘT INSTANCE

### 4.1 Input Tensor [5, 224, 224]

| Kênh | Tên | Mô tả | Giá trị |
|------|-----|-------|--------|
| 0-2 | RGB Image | Ảnh màu gốc | [0, 1] (normalized) |
| 3 | Visible Mask | Phần vật thể nhìn thấy | 0 hoặc 1 |
| 4 | Edge Mask | Ranh giới vật thể | [0, 1] (mềm) |

**Ví dụ:**
```python
# Kênh 0: Ảnh đỏ
tensor[0] = 
  [[0.5, 0.6, 0.4],
   [0.7, 0.8, 0.6],
   ...]

# Kênh 1: Ảnh xanh lá
tensor[1] = 
  [[0.3, 0.4, 0.2],
   ...]

# Kênh 2: Ảnh xanh dương
tensor[2] = 
  [[0.2, 0.3, 0.1],
   ...]

# Kênh 3: Visible Mask (binary)
tensor[3] = 
  [[1, 1, 0],
   [1, 1, 0],
   ...]

# Kênh 4: Edge Mask (soft)
tensor[4] = 
  [[0, 0.5, 0.8],
   [0, 0.3, 0.6],
   ...]
```

### 4.2 Output Label [224, 224]

**Amodal Mask:** Binary mask biểu diễn hình dạng toàn bộ

```python
amodal_label = 
  [[1, 1, 1],
   [1, 1, 1],
   [1, 1, 0],
   ...]
# 1 = vật thể, 0 = background
```

### 4.3 Occlusion Region [224, 224]

**Phần bị che khuất:** = Amodal - Visible

```python
occlusion_region = amodal_label - visible_mask
# Trong training, vùng này được weight cao hơn (5x) trong loss
```

---

## 5. PHÂN LOẠI OCCLUSION

### 5.1 Định nghĩa

**Occlusion Ratio** = (Số pixel bị che) / (Tổng pixel vật thể)

```
occlusion_ratio = np.sum(occlusion_region) / np.sum(amodal_mask)
```

### 5.2 Phân loại

| Loại | Occlusion Ratio | Mô tả | Ví dụ | Số mẫu |
|------|-----------------|-------|-------|--------|
| **Không che** | 0-1% | Vật thể hoàn toàn hiển thị | Một cái bàn ở giữa phòng trống | 9,379 (42.3%) |
| **Che nhẹ** | 1-10% | Vật thể bị che chút chút | Một phần mũ bị che bởi tay | 8,348 (37.7%) |
| **Che vừa** | 10-25% | Phần lớn vật thể nhìn thấy | Nửa trái của người bị che | 3,023 (13.6%) |
| **Che nặng** | >25% | Phần lớn vật thể bị che | Một con chó chỉ thấy đầu | 1,413 (6.4%) |

### 5.3 Sự phân bố

**Training Set:**
```
Không che:       ░░░░░░░░░░  42.3%  →  Dễ học, accuracy cao
Che nhẹ:         ░░░░░░░░░░  37.7%  →  Trung bình
Che vừa:         ░░░░░░░░░░  13.6%  →  Khó hơn
Che nặng:        ░░░░░░░░░░  6.4%   →  Rất khó, cần weight cao
```

---

## 6. CÁC LỚP VẬT THỂ (COCO CATEGORIES)

Cả 91 lớp vật thể của COCO đều có mặt:

### Ví dụ các lớp phổ biến:

| ID | Tên | ID | Tên | ID | Tên |
|----|-----|----|----|----|----|
| 1 | person | 15 | cat | 39 | bottle |
| 2 | bicycle | 16 | dog | 40 | wine glass |
| 3 | car | 17 | horse | 41 | cup |
| 4 | motorcycle | 18 | sheep | 42 | fork |
| 5 | airplane | 19 | cow | 43 | knife |
| 6 | bus | 20 | elephant | 44 | spoon |
| 7 | train | 21 | bear | 45 | bowl |
| 8 | truck | ... | ... | ... | ... |

---

## 7. LỢI ỆI CỦA DESIGN NÀY

### 7.1 Tại sao bóc tách thành instances?

```
❌ Cách cũ:
   1 ảnh = 1 mẫu
   → Nếu ảnh có 5 vật thể → 1 mẫu training
   → Dataset nhỏ, khó học

✅ Cách mới (instances):
   1 ảnh = N mẫu (N = số vật thể)
   → 22K ảnh × 1-2 vật thể/ảnh = 34.9K instances
   → Dataset lớn hơn, dễ học hơn
```

### 7.2 Tại sao cần 5 kênh?

```
RGB (3 kênh)
  ↓
  Thông tin cơ bản: màu sắc, kết cấu
  → Nhưng không biết vùng vật thể ở đâu

+ Visible Mask (1 kênh)
  ↓
  Gợi ý: "Phần này là vật thể"
  → Mô hình biết tập trung vào đâu

+ Edge Mask (1 kênh)
  ↓
  Gợi ý: "Ranh giới trong ảnh"
  → Mô hình biết ranh giới dự đoán nên ở đâu

= 5-kênh input [B, 5, 224, 224]
```

### 7.3 Tại sao weight cao cho occlusion trong loss?

```
Vấn đề:
  - Pixel không che: ~95% của dataset
  - Pixel bị che: ~5% của dataset
  → Class imbalance cực đoan

Giải pháp:
  loss = weighted_bce + dice
  weight[occlusion_region] = 5.0  (5x cao hơn)
  
Tác dụng:
  - Mô hình tập trung vào vùng khó (occlusion)
  - Improve invisible IoU (metric quan trọng)
```

---

## 8. WORKFLOW TOÀN BỘ

```
COCO-Amodal Dataset (JSON + Images)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Dataset Loader                                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Bước 1: Tải annotation + ảnh gốc                       │
│ Bước 2: Vẽ amodal mask, visible mask                   │
│ Bước 3: Tính edge mask, occlusion region               │
│ Bước 4: Data augmentation (Resize, Flip, ...)         │
│ Bước 5: Kết hợp thành 5-kênh input tensor             │
│ Bước 6: Return instance [5, 224, 224] + labels        │
│                                                         │
└─────────────────────────────────────────────────────────┘
    ↓
Batch của instances [B, 5, 224, 224]
    ↓
┌─────────────────────────────────────────────────────────┐
│ Swin-UNet Model                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Encoder: Trích xuất đặc trưng                          │
│ Decoder: Khôi phục độ phân giải                        │
│ Head: Dự đoán amodal mask [B, 1, 224, 224]           │
│                                                         │
└─────────────────────────────────────────────────────────┘
    ↓
Logits [B, 1, 224, 224]
    ↓
┌─────────────────────────────────────────────────────────┐
│ Loss Calculation                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ BCE Loss (weighted):                                    │
│   weight = 1 (không che)                               │
│   weight = 5 (bị che)                                  │
│                                                         │
│ + Dice Loss (balance)                                  │
│                                                         │
│ = Total Loss                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
    ↓
Backpropagation → Update Weights
    ↓
Lặp lại qua tất cả instances trong epoch
```

---

## 9. THỐNG KÊ MẪU

### Ví dụ Instance #1: Người không bị che

```
Input Tensor [5, 224, 224]:
├─ RGB channels: [224×224] chuẩn hóa
├─ Visible mask: Toàn bộ hình dạng người = 1, background = 0
└─ Edge mask: Viền quanh người

Output Label [224, 224]:
├─ Amodal mask: Toàn bộ hình dạng người = 1
└─ Occlusion region: Toàn bộ = 0 (không bị che)

Occlusion Ratio: 0% → Phân loại: "Không che"
```

### Ví dụ Instance #2: Người bị che 15%

```
Input Tensor [5, 224, 224]:
├─ RGB: Ảnh gốc
├─ Visible mask: Phần người nhìn thấy (85%) = 1, phần che = 0
└─ Edge mask: Ranh giới che

Output Label [224, 224]:
├─ Amodal mask: Toàn bộ người (100%) = 1
└─ Occlusion region: Phần bị che (15%) = 1

Occlusion Ratio: 15% → Phân loại: "Che vừa (10-25%)"

Loss tính toán:
├─ Phần nhìn thấy: weight = 1.0
└─ Phần bị che: weight = 5.0  ← Tập trung hơn
```

---

## 10. TÓM TẮT

| Khía cạnh | Chi tiết |
|----------|---------|
| **Instance** | Cặp (ảnh RGB, Visible Mask) |
| **Số instances** | 22,163 training + 12,753 validation |
| **Input** | 5-kênh [RGB, Visible, Edge] |
| **Output** | Amodal mask binary |
| **Classes** | 91 COCO classes |
| **Occlusion** | 0% đến >25% |
| **Phân bố** | ~42% không che, ~57% có che |
| **Loss weight** | 1.0 (không che), 5.0 (che) |

---

**Kết luận:** Việc bóc tách instance theo từng vật thể và sử dụng 5-kênh input giúp mô hình học hiệu quả hơn và tập trung vào vùng occlusion quan trọng.
