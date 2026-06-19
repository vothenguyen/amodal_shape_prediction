cd# 📊 Hướng dẫn Chạy Evaluation Pipeline

## ⚡ Cách nhanh nhất (1 lệnh duy nhất)
```bash
python run_evaluation.py
```
Script này sẽ tự động chạy tất cả 4 bước: Quantitative → Qualitative → Failure Analysis → Ablation Study.

---

## 📋 Chạy từng bước riêng biệt

### Step 1: Quantitative Evaluation (Metrics định lượng)
```bash
cd src
python evaluate.py \
  --img-dir ../data/val2014 \
  --ann-file ../data/annotations/COCO_amodal_val2014.json \
  --checkpoint ../checkpoints/swin_amodal_epoch_30.pth \
  --batch-size 16 \
  --num-workers 0 \
  --output ../results/eval_results.json
```
**Output**: 
- Console: Metrics (mIoU, Dice, Invisible mIoU)
- JSON: `results/eval_results.json` (với per_sample_metrics)

---

### Step 2: Qualitative Evaluation (Best Cases - Top 8 ảnh tốt nhất)
```bash
cd src
python qualitative_eval.py \
  --eval-results ../results/eval_results.json \
  --img-dir ../data/val2014 \
  --ann-file ../data/annotations/COCO_amodal_val2014.json \
  --checkpoint ../checkpoints/swin_amodal_epoch_30.pth \
  --top-k 8 \
  --output ../results/qualitative_best_cases.png
```
**Parameters**:
- `--top-k`: Số lượng ảnh tốt nhất cần hiển thị (mặc định 8)
- `--output`: Đường dẫn file PNG output

**Output**: 
- PNG: `results/qualitative_best_cases.png` (3 cột x 8 hàng: Original → GT → Prediction)

---

### Step 3: Failure Analysis (Worst Cases - Top 5 ca khó nhất)
```bash
cd src
python failure_analysis.py \
  --eval-results ../results/eval_results.json \
  --img-dir ../data/val2014 \
  --ann-file ../data/annotations/COCO_amodal_val2014.json \
  --checkpoint ../checkpoints/swin_amodal_epoch_30.pth \
  --failure-threshold 0.3 \
  --num-worst-show 5 \
  --output ../results/failure_worst_cases.png \
  --save-details \
  --details-output ../results/failure_details.json
```
**Parameters**:
- `--failure-threshold`: Ngưỡng IoU để xác định "ca khó" (mặc định 0.3)
- `--num-worst-show`: Số ca khó nhất cần hiển thị (mặc định 5)
- `--save-details`: Lưu chi tiết phân tích (occlusion %, complexity score)

**Output**: 
- PNG: `results/failure_worst_cases.png`
- JSON: `results/failure_details.json` (với thống kê occlusion, complexity)

---

### Step 4: Ablation Study (So sánh WITH vs WITHOUT Spatial Attention)
```bash
cd src
python ablation_study.py \
  --img-dir ../data/val2014 \
  --ann-file ../data/annotations/COCO_amodal_val2014.json \
  --checkpoint ../checkpoints/swin_amodal_epoch_30.pth \
  --batch-size 16 \
  --num-workers 0 \
  --output ../results/ablation_results.json
```
**Output**: 
- Console: Kết quả so sánh
  ```
  With Spatial Attention:    84.09%
  Without Spatial Attention: 84.07%
  Improvement:               +0.02%
  ```
- JSON: `results/ablation_results.json`

---

## 📁 Cấu trúc Output

```
results/
├── eval_results.json              # Metrics + per_sample_metrics
├── qualitative_best_cases.png     # Top 8 ảnh tốt nhất
├── failure_worst_cases.png        # Top 5 ca khó nhất
├── failure_details.json           # Chi tiết failure cases (occlusion, complexity)
└── ablation_results.json          # Kết quả ablation study
```

---

## 📊 Nội dung trình bày báo cáo

### Phần 1: Metrics Định lượng (1 trang)
Từ `eval_results.json`:
- Overall mIoU: X%
- Dice Coefficient: Y%
- Invisible mIoU: Z%

### Phần 2: Best Cases (1-2 trang)
Từ `qualitative_best_cases.png`:
- Hiển thị hình 3 cột x 8 hàng
- Nhận xét: Model hoạt động rất tốt trên những case...

### Phần 3: Failure Cases (1 trang)
Từ `failure_worst_cases.png` + `failure_details.json`:
- Tỷ lệ ca khó: X% (sample có IoU < 30%)
- 5 ca khó nhất với giải thích:
  - Ca #1: Occlusion 95%, Complexity 0.8 → "Vật thể bị che khuất quá nhiều"
  - Ca #2: Occlusion 20%, Complexity 0.1 → "Vật thể quá nhỏ"
  - Etc.

### Phần 4: Ablation Study (1-2 đoạn)
Từ `ablation_results.json`:
- Bảng so sánh: With/Without Spatial Attention
- Kết luận: "Spatial Attention cải thiện +0.02% (rất nhỏ)"
  - → Có thể Spatial Attention ko cần thiết với kiến trúc này
  - → Hoặc Spatial Attention ko được huấn luyện tốt
  - → Đề xuất: Có thể xem xét các thành phần khác (Skip Connections, Category Embedding, etc.)

---

## ⚙️ Các option phổ biến khác

### Chỉ chạy evaluation trên subset nhỏ (test nhanh)
```bash
cd src
# Giảm batch-size để test nhanh trên ít GPU memory
python evaluate.py --batch-size 4 --num-workers 0 --output ../results/eval_small.json
```

### Tăng số best/worst cases để hiển thị
```bash
# Hiển thị top 16 best cases
python qualitative_eval.py --eval-results ../results/eval_results.json --top-k 16 --output ../results/best_16.png

# Hiển thị 10 worst cases
python failure_analysis.py --eval-results ../results/eval_results.json --num-worst-show 10 --output ../results/worst_10.png
```

---

## 💡 Tips

1. **Lần đầu tiên chạy evaluation**: Dùng `python run_evaluation.py` (tự động hết)
2. **Muốn thử nghiệm thêm**: Chạy từng script riêng với các parameter khác
3. **Nếu GPU memory không đủ**: Giảm `--batch-size` hoặc set `--num-workers 0`
4. **Kết quả ablation study quá nhỏ?** → Có thể Spatial Attention không tác động nhiều trên checkpoint này, hoặc cần train riêng model WITHOUT attention để công bằng
