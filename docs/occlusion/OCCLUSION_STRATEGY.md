# 📊 Chiến lược Tăng mIOU cho Amodal Occlusion Prediction

## 🔍 Vấn đề Hiện Tại

### Phân tích Dataset (Hoàn tất ✅)
- **Training set**: 22,163 mẫu
  - 42.3% không có occlusion (hoặc <1%)
  - 57.7% có occlusion nhưng phần lớn rất nhẹ
  - Median occlusion ratio: 5.4%
  - **Chỉ 4.2% mẫu có occlusion > 75%** ⚠️
  
- **Root cause**: Model học từ dữ liệu dominated bởi mẫu ít che khuất
  - → Học pattern: "Không che khuất → Dự đoán là visible"
  - → Performance trên occluded regions rất thấp

---

## 💡 Giải Pháp Triển Khai (3 Phương Pháp)

### Phương Pháp 1: Tuning Occlusion Weight (5x → 10x, 15x, 20x)
**Nguyên lý**: Tăng penalty cho lỗi trong vùng che khuất

**Formula**:
```
Loss = Weighted BCE + Dice
Weighted BCE = BCE_loss * weight_matrix
weight_matrix[occluded > 0.5] = occlusion_weight (10x, 15x, 20x thay vì 5x)
```

**Ưu điểm**:
- ✅ Đơn giản, không thay đổi architecture
- ✅ Dễ so sánh (A/B test)
- ✅ Có thể tuning smooth qua hyperparameter

**Nhược điểm**:
- ❌ Có thể overfitting trên mẫu occluded nếu weight quá cao
- ❌ Không focus vào hard negatives

---

### Phương Pháp 2: Focal Loss cho Occluded Regions (Hard Negative Mining)
**Nguyên lý**: Tập trung vào những pixel model dự đoán sai (hard negatives)

**Formula**:
```
Focal Loss = -α * (1-p)^γ * log(p)
- γ=2: Penalize hard negatives mạnh hơn easy negatives
- Khi model sai (p≈0 nhưng target=1): (1-p)≈1 → focal_weight lớn
```

**Áp dụng cho occlusion**:
```
focal_weight = (1 - pred_prob)^2
weighted_bce = bce_loss * focal_weight * occlusion_weight_matrix
```

**Ưu điểm**:
- ✅ Tự động focus vào hard cases (pixel mà model sai)
- ✅ Giảm overfitting trên easy negatives
- ✅ Cải thiện precision trên khó khăn

**Nhược điểm**:
- ❌ Hyperparameter tuning phức tạp (α, γ)

---

### Phương Pháp 3: Balanced Sampling (Oversample Occluded Samples)
**Nguyên lý**: Tăng tỷ lệ mẫu occluded trong training → học cân bằng hơn

**Strategy**:
```
1. Tính occlusion_ratio cho mỗi sample
2. Filter samples có occlusion > threshold (e.g., 10%, 25%)
3. Oversample occluded samples (2x, 3x, ...)
4. WeightedRandomSampler → balanced batch composition
```

**Configuration**:
```python
# Option A: Threshold = 10% (44.4% mẫu)
occlusion_threshold = 0.1
oversample_ratio = 2.0  # gấp đôi

# Option B: Threshold = 25% (30.7% mẫu) - CÓ THỂ QUỐC SAM HƠN
occlusion_threshold = 0.25
oversample_ratio = 2.5
```

**Ưu điểm**:
- ✅ Giảm data imbalance rõ ràng
- ✅ Học được pattern của occluded regions
- ✅ Có thể kết hợp với loss tuning

**Nhược điểm**:
- ❌ Dataset nhỏ hơn → training chậm
- ❌ Có thể learn spurious patterns từ oversampling

---

## 🚀 Hướng Dẫn Chạy (Từ Cơ Bản đến Nâng Cao)

### Setup Chuẩn Bị
```bash
cd /path/to/amodal_shape_project

# 1. Phân tích occlusion distribution (nếu chưa chạy)
python src/analyze_occlusion.py
# Output: results/occlusion_analysis_train/ & results/occlusion_analysis_val/
```

### Cách Chạy Training

#### 🔴 Phương Pháp 1: Baseline (Original 5x)
```bash
python src/train_balanced.py \
    --loss-type original \
    --occlusion-weight 5.0 \
    --no-balanced-sampling \
    --epochs 50 \
    --resume-epoch 0
```
**Kì vọng**: Baseline để so sánh

#### 🟡 Phương Pháp 1a: Tuned Weight 10x
```bash
python src/train_balanced.py \
    --loss-type original \
    --occlusion-weight 10.0 \
    --no-balanced-sampling \
    --epochs 50 \
    --resume-epoch 0
```

#### 🟡 Phương Pháp 1b: Tuned Weight 15x (RECOMMENDED)
```bash
python src/train_balanced.py \
    --loss-type original \
    --occlusion-weight 15.0 \
    --no-balanced-sampling \
    --epochs 50 \
    --resume-epoch 0
```

#### 🟡 Phương Pháp 1c: Tuned Weight 20x
```bash
python src/train_balanced.py \
    --loss-type original \
    --occlusion-weight 20.0 \
    --no-balanced-sampling \
    --epochs 50 \
    --resume-epoch 0
```

#### 🟢 Phương Pháp 2: Focal Loss (Hard Negative Mining)
```bash
python src/train_balanced.py \
    --loss-type focal \
    --occlusion-weight 10.0 \
    --focal-gamma 2.0 \
    --no-balanced-sampling \
    --epochs 50 \
    --resume-epoch 0
```

#### 🔵 Phương Pháp 3: Balanced Sampling (10% threshold, 2x oversample)
```bash
python src/train_balanced.py \
    --loss-type original \
    --occlusion-weight 10.0 \
    --use-balanced-sampling \
    --occlusion-threshold 0.1 \
    --oversample-ratio 2.0 \
    --epochs 50 \
    --resume-epoch 0
```

#### 🟣 Phương Pháp 3 (Nâng cao): Balanced + Focal (BEST COMBO)
```bash
python src/train_balanced.py \
    --loss-type combo \
    --occlusion-weight 15.0 \
    --focal-gamma 2.0 \
    --use-balanced-sampling \
    --occlusion-threshold 0.1 \
    --oversample-ratio 2.0 \
    --epochs 50 \
    --resume-epoch 0
```

---

## 📈 So Sánh & Evaluation

### Chạy Evaluation trên Validation Set
```bash
# Baseline
python src/evaluate.py \
    --checkpoint checkpoints/swin_amodal_epoch_50.pth \
    --output results/eval_baseline_5x.json

# 15x weight
python src/evaluate.py \
    --checkpoint checkpoints/swin_amodal_epoch_50.pth \
    --output results/eval_tuned_15x.json

# Balanced + Focal
python src/evaluate.py \
    --checkpoint checkpoints/swin_amodal_epoch_50.pth \
    --output results/eval_balanced_focal.json
```

### So Sánh Metrics
Các metrics cần so sánh:
1. **Overall mIoU**: Tổng độ chính xác
2. **Invisible mIoU**: Độ chính xác chỉ trên vùng che khuất (QUAN TRỌNG!)
3. **Dice Coefficient**: Measure khác của segmentation quality

**Kỳ vọng**:
- Overall mIoU: +2-5% improvement
- Invisible mIoU: +5-15% improvement (CHÍNH LÀ MỤC TIÊU!)

---

## 🎯 Giải Thích Chi Tiết cho Từng Phương Pháp

### Tại Sao Balanced Sampling + Focal Loss?

1. **Balanced Sampling**:
   - Dataset ban đầu: 43% non-occluded, 57% occluded (nhưng 80% trong đó <25%)
   - Filter threshold 10%: 44.4% mẫu được giữ lại (23 mẫu → 10 mẫu đã occluded 10%+)
   - Oversample 2x: Trong 1 epoch, 2/3 batch là occluded, 1/3 là non-occluded
   - → Model học pattern occluded region tốt hơn

2. **Focal Loss (γ=2)**:
   - Ví dụ: Pixel trong vùng occluded
     - Model predict p=0.1 (sai), target=1
     - BCE_loss = log(1/0.1) ≈ 2.3
     - focal_weight = (1-0.1)^2 = 0.81 ← tăng penalty
     - weighted_loss = 2.3 × 0.81 ≈ 1.86
   - Model tập trung học những pixel khó (p < 0.5)

3. **Occlusion Weight 15x**:
   - Occluded pixel: loss × 15
   - Non-occluded pixel: loss × 1
   - 15:1 ratio force model priority occluded regions

4. **Kết Hợp 3 chiến lược**:
   - Balanced sampling: cấu trúc tốt (data composition)
   - Occlusion weight: priority (loss weighting)
   - Focal loss: hard negative mining (automatic focus)
   - → **Tối ưu từ 3 góc độ**

---

## ⚡ Troubleshooting

### Nếu loss quá cao (exploding):
```bash
# Giảm occlusion weight
python src/train_balanced.py --occlusion-weight 10.0 --loss-type original

# Hoặc giảm learning rate
python src/train_balanced.py --learning-rate 5e-5
```

### Nếu training không hội tụ:
```bash
# Kiểm tra balanced sampler
python -c "from src.advanced_loss import create_balanced_dataloader; ..."

# Hoặc tắt balanced sampling
python src/train_balanced.py --no-balanced-sampling
```

### Nếu checkpoint không load được:
```bash
# Kiểm tra checkpoint path
ls -la checkpoints/

# Hoặc reset
python src/train_balanced.py --resume-epoch 0
```

---

## 📝 Tóm Tắt Chiến Lược

| Phương Pháp | Loss Type | Occlusion Weight | Balanced? | Improvement |
|---|---|---|---|---|
| **Baseline** | Original | 5x | ❌ | - |
| **Tuned 10x** | Original | 10x | ❌ | +1-2% |
| **Tuned 15x** | Original | 15x | ❌ | +2-3% |
| **Focal** | Focal | 10x | ❌ | +3-4% |
| **Balanced 10%** | Original | 10x | ✅ | +4-6% |
| **Best Combo** | Combo | 15x | ✅ 10% | **+7-12%** ⭐ |

---

## 🔄 Tiếp Theo Có Thể Thử

1. **Data Augmentation**:
   - Thêm specific augmentation cho occluded regions
   - Tạo synthetic occlusion

2. **Architecture**:
   - Add attention mechanism specifically for occlusion
   - Separate head cho visible vs occluded prediction

3. **Post-processing**:
   - Morphological operations
   - CRF (Conditional Random Fields)

---

**Created**: 2025-04-22
**Status**: Ready for implementation ✅
