# Amodal Shape Project

Báo cáo dự án: Phân đoạn Amodal với kiến trúc Swin Transformer + U-Net.

## 1. Mục tiêu
- Dự án xây dựng mô hình phân đoạn **amodal mask**: dự đoán vùng vật thể đầy đủ ngay cả khi bị che khuất.
- Input: ảnh RGB + kênh `visible` (phần không bị che khuất).
- Output: amodal mask nhị phân (1 = vật thể, 0 = nền/che khuất).

## 2. Cấu trúc thư mục
- `src/`:
  - `dataset.py`: lớp `AmodalDataset` (cố lấy từng region làm 1 sample), tạo `amodal_mask` + `visible_mask`, biến đổi, trả về tensor 4 kênh (RGB + visible) và nhãn amodal.
  - `model.py`: `AmodalSwinUNet` (encoder Swin-Tiny + decoder U-Net với skip connections). Mô hình ra 1 kênh logits.
  - `train.py`: pipeline huấn luyện, loss `BCEDiceLoss` (BCE + Dice), AdamW.
  - `test_model.py`: thử nghiệm batch inference với checkpoint, hiển thị grid: RGB / Visible / Dự đoán / Ground truth.
- `checkpoints/`: weights đã lưu (epoch 1..24).
- `data/`: dữ liệu ảnh `train2014`, `val2014`, annotation COCO amodal.
- `notebooks/`: ví dụ khám phá, trực quan dữ liệu và gọi model.

## 3. Dữ liệu
- `data/annotations/COCO_amodal_train2014.json` (tương tự COCOA format, annotations chứa `regions` và trường `segmentation` + `order`).
- `AmodalDataset`:
  - mỗi viễn cảnh `ann_id` -> nhiều `region`, tạo sample cho từng region.
  - amodal mask từ region segmentation.
  - visible mask: xóa phần bị che khuất dựa vào `order` (vật thấp hơn che lên).
  - resize 224x224 (đồng bộ image + mask thông qua albumentations).
  - đầu ra: input [4, 224, 224], target [224, 224].

## 4. Mô hình
- Encoder: `timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)`
- Sửa patch_embed đầu vào thành 4 channel.
- Decoder sử dụng 3 khối `UpBlock` và bước `nn.Upsample(scale_factor=4)`.
- Head: `Conv2d(64,1,1)` (logits).
- Kích thước output: `[B, 1, 224, 224]`.

## 5. Huấn luyện
- Loss: `BCEDiceLoss` (BCEWithLogits + Dice)
- Optimizer: `AdamW`, lr=1e-4
- Batch size 4, epochs 50.
- Lưu checkpoints tại `../checkpoints/swin_amodal_epoch_{epoch}.pth`.

## 6. Inference
- `batch_test()` tải `swin_amodal_epoch_24.pth`
- Dự đoán với `sigmoid` + threshold 0.5
- In ra 4 bảng: RGB / visible / prediction / ground truth.

## 7. Yêu cầu môi trường
- Python 3.8+ (hoặc tương đương với torch/timm)
- PyTorch
- timm
- albumentations
- opencv-python
- pycocotools
- matplotlib
- tqdm

## 8. Chạy nhanh
1. Cài dependencies: `pip install -r requirements.txt` (nếu có) hoặc `pip install torch timm albumentations opencv-python pycocotools matplotlib tqdm`
2. Huấn luyện: `python src/train.py`
3. Test: `python src/test_model.py`
4. Notebook khám phá: `notebooks/explore_data.ipynb`.

## 9. Quy trình đánh giá toàn diện (Evaluation Pipeline)
Để báo cáo trở thành "siêu phẩm", cần 3 phần evaluation:

### A. Đánh giá Định lượng (Quantitative)
Metrics định lượng: mIoU, Dice, Invisible mIoU.
```bash
python src/evaluate.py --img-dir ../data/val2014 --ann-file ../data/annotations/COCO_amodal_val2014.json --checkpoint ../checkpoints/swin_amodal_epoch_30.pth --output results/eval_results.json
```

### B. Đánh giá Định tính (Qualitative - Show hình ảnh)
Hiển thị top-8 ảnh có IoU cao nhất (Original → Ground Truth → Prediction).
```bash
python src/qualitative_eval.py --eval-results results/eval_results.json --top-k 8 --output results/qualitative_best.png
```

### C. Phân tích Ca khó (Failure Analysis)
Tìm 5 ca khó nhất (IoU < 30%) và giải thích tại sao model sai.
```bash
python src/failure_analysis.py --eval-results results/eval_results.json --threshold 0.3 --worst-k 5 --output results/failure_cases.png
```

### D. Ablation Study
Chứng minh sức đóng góp của Spatial Attention: tăng +X% hiệu suất.
```bash
python src/ablation_study.py --checkpoint ../checkpoints/swin_amodal_epoch_30.pth --output results/ablation_results.json
```

### Nội dung trình bày
- Metrics định lượng (1 bảng)
- Hình ảnh best cases (1-2 trang)
- Hình ảnh failure cases + giải thích (1 trang)
- Ablation Study results (1-2 đoạn)

Chi tiết xem [`EVALUATION.md`](EVALUATION.md)

## 10. Ghi chú thêm
- Code dùng nhiều comment tiếng Việt giúp hiểu nhanh flow.
- Mô hình có thể mở rộng bằng: augment thêm, validation set, learning rate scheduler, early stopping, metrics IOU/F1.
