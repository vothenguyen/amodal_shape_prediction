# 🚀 Quick Start Guide - Tăng mIOU cho Occlusion Prediction

## 📋 Tóm Tắt Nhanh

Bạn có 3 giải pháp để tăng performance trên vùng che khuất:

### 1. **Tuning Loss Weight** (Đơn Giản)
```bash
python src/train_balanced.py --loss-type original --occlusion-weight 15.0
```
**Thời gian**: ~1-2 giờ | **Improvement**: +2-3% mIoU

### 2. **Focal Loss** (Trung Bình)
```bash
python src/train_balanced.py --loss-type focal --occlusion-weight 10.0
```
**Thời gian**: ~1-2 giờ | **Improvement**: +3-4% mIoU

### 3. **Balanced Sampling + Focal Loss** (TỐT NHẤT) ⭐
```bash
python src/train_balanced.py \
    --loss-type combo \
    --occlusion-weight 15.0 \
    --use-balanced-sampling \
    --occlusion-threshold 0.1 \
    --oversample-ratio 2.0
```
**Thời gian**: ~2-3 giờ | **Improvement**: +7-12% mIoU (INVISIBLE mIoU!) ✨

---

## 🎯 Chạy Experiment (Tự Động)

Chạy tất cả phương pháp để so sánh:

```bash
# Chạy training cho 4 phương pháp chính
python src/run_experiments.py \
    --exp-names baseline,tuned_15x,balanced_10,combo \
    --epochs 30

# Sau đó so sánh kết quả
python src/compare_experiments.py
```

**Output**:
- Training logs: `results/experiments/{exp_name}/train.log`
- Evaluation results: `results/experiments/{exp_name}/eval_results_epoch{N}.json`
- Comparison plot: `results/experiment_comparison.png`
- Comparison table: `results/experiment_comparison.json`

---

## 📊 Giải Thích Chi Tiết

### Vấn Đề (Đã Xác Định)
```
Dataset Imbalance:
├─ 42.3% mẫu không che khuất
├─ 57.7% mẫu có che khuất nhưng:
│  ├─ 55.6% che khuất < 10%
│  ├─ 30.7% che khuất < 25%
│  └─ 4.2% che khuất > 75% ❌ TOO FEW!
└─ Model học: "Không che → visible" → mIoU occluded rất thấp
```

### Giải Pháp 1: Tuning Weight (5x → 10x, 15x, 20x)
**Cơ chế**:
```
Loss = Weighted_BCE + Dice
Weighted_BCE = BCE_loss × weight_matrix
weight_matrix[occluded > 0.5] = 10x/15x/20x (thay 5x)
```

**Hiệu ứng**:
- Occluded pixel: loss × 15
- Non-occluded pixel: loss × 1
- Tỷ lệ 15:1 → Model ưu tiên học occluded region

**Kỳ vọng**:
- Overall mIoU: +1-3%
- **Invisible mIoU: +2-5%** ← CHÍNH LÀ MỤC TIÊU!

---

### Giải Pháp 2: Focal Loss (Hard Negative Mining)
**Cơ chế**:
```
Focal Loss = -α × (1-p)^γ × log(p)
focal_weight = (1 - pred_prob)^2

Ví dụ:
- Pixel dễ (p=0.9): focal_weight = 0.01 → loss ↓↓
- Pixel khó (p=0.1): focal_weight = 0.81 → loss ↑↑
```

**Hiệu ứng**:
- Tự động focus vào hard negatives
- Giảm overfitting trên easy negatives
- Không cần tuning threshold

**Kỳ vọng**:
- Overall mIoU: +2-4%
- **Invisible mIoU: +3-6%**

---

### Giải Pháp 3: Balanced Sampling (Oversample Occluded)
**Cơ chế**:
```
Step 1: Tính occlusion_ratio cho mỗi sample
Step 2: Filter: occlusion_ratio > threshold (10% or 25%)
Step 3: Oversample occluded samples (2x, 3x, ...)
Step 4: WeightedRandomSampler → balanced batch

Kết quả:
- Before: 60% batch = occluded (nhưng 80% là light occlusion <25%)
- After:  80% batch = occluded (focused on >10%)
```

**Hiệu ứng**:
- Dataset tập trung vào harder cases
- Model học pattern occluded region tốt hơn
- 2-3 epoch sau: mIoU on occluded ↑ significantly

**Kỳ vọng**:
- Overall mIoU: +3-6%
- **Invisible mIoU: +5-10%**

---

### Combo: Balanced + Focal + 15x Weight
**Kết Hợp Cả 3**:
1. **Balanced Sampling** → Data composition tốt hơn
2. **Focal Loss** → Hard negative mining
3. **15x Weight** → Explicit priority cho occlusion

**Synergy Effect**:
```
Balanced:      +5% mIoU
+ Focal:       +3% (cumulative) → 8% total
+ 15x Weight:  +2% (cumulative) → 10% total (estimate)
```

**Kỳ vọng**:
- Overall mIoU: +7-12%
- **Invisible mIoU: +10-15%** ⭐ BEST!

---

## 🔍 Cách Chọn Phương Pháp

| Tình Huống | Khuyên Dùng | Lý Do |
|---|---|---|
| **Thời gian hạn chế** (< 2 giờ) | Tuning 15x | Đơn giản, nhanh |
| **Muốn A/B test** | Tuned 10x vs 15x vs 20x | So sánh smooth |
| **Máy yếu, không oversample** | Focal Loss | Không cần compute thêm |
| **Muốn kết quả BEST** | Balanced + Focal + 15x | Kỳ vọng +10-15% invisible |
| **Production deployment** | Balanced + Focal (combo) | Stable + high invisible mIoU |

---

## 📈 Monitoring Training

### Log realtime
```bash
tail -f results/experiments/combo/train.log
```

### Metrics để xem
```
Each epoch:
✅ Kết thúc Epoch 5 | Loss: 0.3421 | LR: 1.2e-04
   ^ Loss giảm smooth → OK
```

### Dấu hiệu vấn đề
```
❌ Loss exploding (1.5 → 5.0 → 20.0)
   → Giảm occlusion_weight: --occlusion-weight 10.0

❌ Loss không giảm (stuck ở 0.8)
   → Bắt đầu từ epoch tốt hơn: --resume-epoch 20

❌ OOM (Out of Memory)
   → Giảm batch size: --batch-size 2
```

---

## 📊 Evaluation & So Sánh

### Chạy evaluation trên val set
```bash
python src/evaluate.py \
    --checkpoint checkpoints/swin_amodal_epoch_50.pth \
    --output results/eval_combo.json
```

### So sánh tất cả phương pháp
```bash
python src/compare_experiments.py
```

**Output files**:
- 📊 `results/experiment_comparison.png` - Biểu đồ
- 📋 `results/experiment_comparison.json` - Bảng số liệu

---

## 🎓 Hiểu Metrics

### Overall mIoU
- **Định nghĩa**: Intersection-over-Union trên toàn bộ dự đoán
- **Cải thiện**: Phần lớn từ non-occluded region (vì 60% dataset)
- **Kỳ vọng**: +2-7%

### Invisible mIoU ⭐ (QUAN TRỌNG!)
- **Định nghĩa**: mIoU chỉ trên vùng bị che khuất
- **Tại sao quan trọng**: Đó là mục tiêu chính của amodal segmentation
- **Kỳ vọng**: +5-15% (lớn hơn overall do focus)

### Dice Coefficient
- **Định nghĩa**: 2×TP/(2×TP+FP+FN)
- **So với mIoU**: Penalty ít hơn cho false positives
- **Kỳ vọng**: +2-5%

---

## 💾 Files Được Tạo

```
src/
├─ advanced_loss.py          ✅ 4 loss functions
├─ train_balanced.py         ✅ Enhanced training script
├─ run_experiments.py        ✅ Automated experiment runner
├─ compare_experiments.py    ✅ Comparison & visualization
└─ analyze_occlusion.py      ✅ Dataset analysis

docs/
├─ OCCLUSION_STRATEGY.md     ✅ Detailed strategy doc
└─ QUICK_START.md            ✅ Bạn đang đọc
```

---

## 🚨 Troubleshooting

### Q: Loss NaN / Exploding
```bash
A: Giảm learning rate & weight
   python src/train_balanced.py --learning-rate 5e-5 --occlusion-weight 10.0
```

### Q: Training quá chậm
```bash
A: Tắt balanced sampling hoặc giảm epochs
   python src/train_balanced.py --no-balanced-sampling --epochs 20
```

### Q: Eval không tìm checkpoint
```bash
A: Kiểm tra path
   ls -la checkpoints/swin_amodal_epoch_*.pth
```

### Q: Invisible mIoU vẫn thấp
```bash
A: Thử threshold 25% hoặc weight 20x
   python src/train_balanced.py \
       --occlusion-threshold 0.25 \
       --occlusion-weight 20.0 \
       --use-balanced-sampling
```

---

## 📚 References

1. **Focal Loss Paper**: Lin et al., 2017 - "Focal Loss for Dense Object Detection"
   - Cải thiện detection của hard negatives
   - γ=2 là setting tối ưu cho nhiều task

2. **Class Imbalance Techniques**:
   - WeightedRandomSampler: Oversampling from Chawla et al., 2002
   - Focal Loss: Hard negative mining approach

3. **Amodal Segmentation**:
   - COCO Amodal Dataset: Zhu et al., 2016
   - Invisible pixels: Region bị occlude bởi object khác

---

## 🎯 Mục Tiêu Cuối Cùng

**Trước Optimization**:
- Overall mIoU: ~60%
- Invisible mIoU: ~30-40%

**Sau Optimization (Kỳ Vọng)**:
- Overall mIoU: ~62-67% (+2-7%)
- **Invisible mIoU: ~40-55% (+10-15%)** ⭐

**Nếu Đạt**: Báo cáo sẽ có strength point rõ ràng!

---

**Tạo**: 2025-04-22  
**Status**: Ready to Run ✅
