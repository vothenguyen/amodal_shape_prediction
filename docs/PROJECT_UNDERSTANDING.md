# 📖 HIỂU BIẾT VỀ DỰ ÁN — Amodal Shape Prediction

> Tài liệu tóm tắt Phase 1 — Agent đọc hiểu toàn bộ context dự án trước khi mở rộng sang D2SA.

---

## 1. README.md gốc — Kiến trúc & cấu trúc thư mục

- **Pipeline 2-Stage:**
  - Stage 1: SAM 2.1 → Visible Mask từ point prompt (zero-shot).
  - Stage 2: Swin-UNet 5 kênh (RGB + Visible Mask + Edge Mask) + Category Embedding → Amodal Mask `[B, 1, 224, 224]`.
- **Dữ liệu:** COCOA format (`regions`, `segmentation`, `order`), resize về `224×224` bằng `albumentations`.
- **Cấu trúc thư mục:**
  ```
  ├── app.py                  # Gradio Web UI (dùng SAM 2.1 + Swin-UNet)
  ├── requirements.txt
  ├── assets/                 # Hình minh hoạ (Table.png, figures/)
  ├── checkpoints/            # Weights (sam2.1_b.pt, swin_amodal_epoch_30.pth) — nằm trong .gitignore
  ├── data/                   # Dữ liệu (nằm trong .gitignore)
  ├── docs/                   # Tài liệu báo cáo
  ├── results/                # Kết quả JSON/PNG (nằm trong .gitignore)
  └── scripts/
      ├── model.py            # Cấu hình chính Row 4
      ├── dataset.py          # AmodalDataset cho COCOA
      ├── train.py            # Train Row 4 cho COCOA
      ├── evaluate.py         # Đánh giá single-image
      ├── train_lambda.py     # Thử nghiệm nhiều occlusion_weight (1,3,5,7,10)
      ├── train_kins.ipynb    # Notebook Colab train KINS (6 config)
      └── other_config_cocoa/ # 5 biến thể ablation (Row 1,2,3,5,6)
  ```

---

## 2. scripts/README.md — 6 cấu hình ablation

> **Lưu ý:** File `scripts/README.md` **không tồn tại** trong repo hiện tại. Thông tin về 6 cấu hình ablation được mô tả trong notebook `train_kins.ipynb` và qua cấu trúc thư mục `other_config_cocoa/`.

### Bảng tổng hợp 6 cấu hình (Table V — Ablation Study):

| Row | Tên thư mục / vị trí | Loss | Edge Mask | Cat Embedding | Spatial Attn | Input Channels | forward() nhận class_ids? |
|-----|----------------------|------|-----------|---------------|--------------|----------------|--------------------------|
| **1** | `other_config_cocoa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/` | BCE thường | ❌ | ❌ | ❌ | 4 (RGB+Vis) | ❌ `forward(x)` |
| **2** | `other_config_cocoa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/` | BCE thường | ✅ | ❌ | ❌ | 5 (RGB+Vis+Edge) | ❌ `forward(x)` |
| **3** | `other_config_cocoa/amodal-shape-prediction-no-spatial-no-embeding/` | Occ-Aware (5x) | ✅ | ❌ | ❌ | 5 (RGB+Vis+Edge) | ❌ `forward(x)` |
| **4** | `scripts/` (gốc — `model.py` + `train.py`) | Occ-Aware (5x) | ✅ | ✅ (91 lớp) | ❌ | 5 (RGB+Vis+Edge) | ✅ `forward(x, class_ids)` |
| **5** | `other_config_cocoa/amodal_shape_prediction_no_embeding/` | Occ-Aware (5x) | ✅ | ❌ | ✅ | 5 (RGB+Vis+Edge) | ❌ `forward(x)` |
| **6** | `other_config_cocoa/amodal_shape_prediction_full_config/` | Occ-Aware (5x) | ✅ | ✅ (91 lớp) | ✅ | 5 (RGB+Vis+Edge) | ✅ `forward(x, class_ids)` |

---

## 3. scripts/model.py — Kiến trúc Swin-UNet

- **Encoder:** `timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)`.
  - Patch embedding được sửa để nhận 5 kênh (RGB + Vis + Edge). Trọng số 3 kênh RGB được giữ từ pretrained, 2 kênh bổ sung khởi tạo = 0.
- **Category Embedding:** `nn.Embedding(num_classes, 768)` → cộng vào bottleneck `[B, 768, 7, 7]` bằng broadcasting.
- **Decoder:** 3 UpBlock (ConvTranspose2d + DoubleConv + skip connection) → `up_final` (Upsample ×4 + Conv 96→64) → `final_conv` (Conv 64→1).
- **Output:** logits `[B, 1, 224, 224]` (chưa qua sigmoid).
- **`num_classes` đã là tham số hóa** (default=91), **không hard-code**. ✅ Điều này có nghĩa D2SA chỉ cần truyền `num_classes=<số lớp D2S>` khi khởi tạo model.

---

## 4. scripts/dataset.py — Logic đọc COCOA

- **Class:** `AmodalDataset(img_dir, ann_file, transform)`.
- **Bóc tách instances:** Duyệt qua `self.coco.anns`, mỗi annotation có `regions` → mỗi region = 1 mẫu huấn luyện. Lưu `(ann_id, region_idx)`.
- **Amodal Mask:** Vẽ polygon từ `region["segmentation"]` bằng `cv2.fillPoly`.
- **Visible Mask:** Copy từ amodal mask → xoá (tô đen) phần bị che bởi các vật thể có `order` nhỏ hơn (phía trước).
- **Augmentation:** `albumentations.Compose([Resize, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast])` — áp dụng đồng bộ trên image + cả 2 mask.
- **Edge Mask:** `cv2.dilate(visible, kernel5×5) - cv2.erode(visible, kernel5×5)` → ranh giới visible mask.
- **Output:** `(input_tensor[5,H,W], amodal_tensor[H,W], occluded_region[H,W], cat_id)`.
- **⚠️ Điểm quan trọng cho D2SA:** Dataset class này dùng COCOA-specific logic (field `regions`, `order` để suy visible mask). D2SA có annotation format khác (COCO instance segmentation style, với amodal+visible mask riêng) → **cần viết dataset class riêng**.

---

## 5. scripts/train.py — Vòng lặp train (Row 4 — cấu hình chính)

- **Hyperparameters:** `BATCH_SIZE=4`, `ACCUMULATION_STEPS=4` (effective batch=16), `EPOCHS=30`, `LR=1e-4`.
- **Optimizer:** `AdamW`.
- **Scheduler:** `CosineAnnealingLR(T_max=EPOCHS)`.
- **Loss:** `OcclusionAwareLoss(occlusion_weight=5.0)` = weighted BCE (5x cho vùng bị che) + Dice loss.
- **Checkpoint:** lưu `model.state_dict()` mỗi epoch tại `../checkpoints/swin_amodal_epoch_{epoch}.pth`.
- **Data path:** `img_dir="../data/train2014"`, `ann_file="../data/annotations/COCO_amodal_train2014.json"` — đường dẫn tương đối từ `scripts/`.
- **RESUME_EPOCH:** Hỗ trợ resume nhưng hard-code = 18 (lần cuối train COCOA). Cần đặt lại = 0 cho D2SA.

---

## 6. scripts/evaluate.py — Đánh giá

- **Chế độ:** Single-image (batch_size=1 cố định).
- **Metrics:** mIoU, Dice, Precision, Recall (trên toàn bộ amodal mask) + Invisible mIoU (chỉ vùng bị che).
- **Model load:** Hỗ trợ bóc `_orig_mod.` (từ `torch.compile`) và `model_state_dict` key.
- **Output:** JSON tại `results/per_image_eval.json` + in console.
- **CLI args:** `--img-dir`, `--ann-file`, `--checkpoint`, `--num-workers`, `--resize`, `--threshold`, `--device`, `--output`.
- **⚠️ Lưu ý:** `AmodalSwinUNet(num_classes=91)` hard-code ở dòng 110 → cần sửa/tham số hoá khi evaluate D2SA.

---

## 7. scripts/train_kins.ipynb — KINS

- **Chạy trên Colab** (T4 GPU), pull code từ GitHub nhánh `2.-spartial_attention`.
- **Dữ liệu KINS:** Tải ảnh KITTI từ Kaggle (`anupammajhi/kitti-2d-object-detection`), nhãn KINS từ Kaggle (`ryanthenguyen/kins-data`).
- **Dataset class riêng:** `dataset_kins.KINSDataset` (import `from dataset_kins import KINSDataset`) — **khác hoàn toàn** với `dataset.py` (COCOA). Annotation format KINS cũng dùng `regions` nhưng cấu trúc khác COCOA.
- **num_classes KINS:** Notebook dùng `AmodalSwinUNet()` (default 91). KINS chỉ có ~7 lớp nhưng vẫn dùng embedding table 91 lớp (không tối ưu nhưng vẫn chạy được vì class_id < 91).
- **Chạy 6 cấu hình ablation:** Cả 6 config được chạy tuần tự trong 1 notebook (mỗi config là 1 block code cell), không tách file riêng.
- **Pattern khác COCOA:** KINS dùng 1 notebook duy nhất, COCOA dùng thư mục riêng cho mỗi config.

---

## 8. scripts/other_config_cocoa/ — Tổ chức ablation code cho COCOA

### Pattern tổ chức:
- Mỗi biến thể ablation là **1 thư mục riêng**, chứa **3 file độc lập:** `model.py`, `train.py`, `evaluate.py`.
- Mỗi thư mục là một **bản sao toàn bộ** (không import chung), chỉ khác nhau ở:
  - `model.py`: có/không `SpatialAttention`, có/không `category_emb`, input 4 hay 5 kênh.
  - `train.py`: dùng `OcclusionAwareLoss` hay `BCEWithLogitsLoss`, có/không truyền `class_ids`.
  - `evaluate.py`: tương ứng model.

### 5 thư mục (cho Row 1, 2, 3, 5, 6):
```
other_config_cocoa/
├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce/  # Row 1
├── amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/    # Row 2
├── amodal-shape-prediction-no-spatial-no-embeding/                     # Row 3
├── amodal_shape_prediction_no_embeding/                                # Row 5
└── amodal_shape_prediction_full_config/                                # Row 6
```
Row 4 (cấu hình chính) nằm ở `scripts/` gốc (`model.py`, `train.py`, `evaluate.py`).

### ⚠️ Quyết định cho D2SA:
→ **D2SA nên dùng cùng pattern COCOA:** tạo `scripts/other_config_d2sa/` với cấu trúc tương tự, mỗi config 1 thư mục chứa 3 file. Config chính (Row 4) có thể đặt ở `scripts/` gốc (thêm file `train_d2sa.py`, `dataset_d2sa.py`) hoặc trong `other_config_d2sa/`.

### ⚠️ Lưu ý về dataset.py:
- Các thư mục ablation của COCOA **không chứa `dataset.py`** — chúng import từ `scripts/dataset.py` (hoặc dùng relative import `from dataset import AmodalDataset`). Điều này nghĩa là dataset class được tái sử dụng, chỉ model/train/evaluate thay đổi.
- D2SA cần 1 file `dataset_d2sa.py` riêng (vì annotation format khác), và tất cả 6 config sẽ import từ file này.

---

## 9. Các file khác

### `app.py`
- Gradio Web UI demo pipeline SAM 2.1 + Swin-UNet.
- Hard-code 91 lớp COCO, import `scripts.model.AmodalSwinUNet`.
- **Không cần sửa** cho D2SA (app chỉ demo COCOA).

### `requirements.txt`
```
albumentations==2.0.8, gradio==6.13.0, matplotlib==3.10.9, numpy==2.4.4,
opencv_python==4.13.0.92, opencv_python_headless==4.13.0.92,
pycocotools==2.0.11, timm==1.0.26, torch==2.9.1, torchvision==0.24.1,
tqdm==4.67.1, ultralytics==8.4.41
```
- Không cần thêm thư viện mới cho D2SA (các thư viện cần thiết đã có đủ).

### `.gitignore`
- Đã bỏ qua: `data/`, `checkpoints/*.pt`, `*.pth`, `results/`, `*.json` (ngoại trừ `COCO_amodal_val2014.json`).

### `temp.py` — File scratch/thử nghiệm
- Script demo nhanh: load model → inference 1 ảnh → vẽ overlay → lưu PNG.
- **Là file rác/thử nghiệm**, không thuộc pipeline chính. Đề xuất xoá ở Phase 2 (sau khi được xác nhận).

### `scripts/train_lambda.py`
- Script thử nghiệm nhiều `occlusion_weight` (1,3,5,7,10), mỗi weight train 10 epoch.
- Dùng AMP (mixed precision), batch_size=64 (cho A100).
- **Là file thử nghiệm riêng**, không thuộc 6 cấu hình ablation chính.

---

## 10. Checkpoints

- Thư mục `checkpoints/` **không có trong bản giải nén** (nằm trong `.gitignore`).
- Checkpoint COCOA/KINS được lưu bên ngoài repo (Google Drive khi train trên Colab).

---

## ✅ CHECKLIST TÓM TẮT

| # | Hạng mục | Trạng thái |
|---|---------|-----------|
| 1 | README.md gốc | ✅ Đọc xong |
| 2 | scripts/README.md | ⚠️ **Không tồn tại** — thông tin ablation nằm trong notebook KINS |
| 3 | model.py | ✅ `num_classes` đã tham số hoá (default=91) |
| 4 | dataset.py | ✅ Logic COCOA-specific (regions, order) → cần viết riêng cho D2SA |
| 5 | train.py | ✅ Hyperparams rõ ràng, giữ nguyên cho D2SA |
| 6 | evaluate.py | ✅ `num_classes=91` cần tham số hoá |
| 7 | train_kins.ipynb | ✅ KINS dùng 1 notebook, dataset class riêng (`dataset_kins`) |
| 8 | other_config_cocoa/ | ✅ Mỗi config = 1 thư mục (3 file), D2SA sẽ mirror pattern này |
| 9 | app.py, temp.py, .gitignore | ✅ `temp.py` là rác, app.py không cần sửa |
| 10 | Checkpoints | ✅ Không có trong repo, lưu ngoài |

---

## ⚠️ ĐIỂM CẦN LƯU Ý / CÂU HỎI

1. **D2SA annotation format:** ✅ **ĐÃ XÁC NHẬN** — COCO-style JSON, nhưng **khác hoàn toàn COCOA**.
2. **Số lớp D2SA:** ✅ **Chính xác 60 lớp** (category_id 1–60, liên tục, không thiếu), cả train và val đều phủ đủ 60 lớp. → `num_classes=60`.
3. **evaluate.py hard-code `num_classes=91`:** Cần sửa thành tham số CLI (default=91 giữ nguyên cho COCOA). D2SA truyền `--num-classes 60`.
4. **Pattern D2SA:** Sẽ dùng pattern COCOA (thư mục riêng cho mỗi config) thay vì pattern KINS (1 notebook).

---

## 📊 D2SA ANNOTATION FORMAT — Kết quả kiểm tra thực tế (Phase 3)

### So sánh D2SA vs. COCOA

| Đặc điểm | COCOA | D2SA |
|----------|-------|------|
| **Cấu trúc** | `regions` nested trong annotation, mỗi region có `segmentation` polygon | Flat COCO-style: mỗi annotation = 1 instance |
| **Amodal mask** | Polygon trong `region["segmentation"]` | **RLE** trong `annotation["segmentation"]` |
| **Visible mask** | Tự suy từ `order` (trừ vùng bị che) | **Trực tiếp** trong `annotation["visible_mask"]` (RLE) |
| **Invisible mask** | Không có trực tiếp, phải tự tính | **Trực tiếp** trong `annotation["invisible_mask"]` (RLE, optional) |
| **Occlusion info** | `order` field (depth ordering) | `occlude_rate` (float) + `occl_depth` (int) |
| **Mask encoding** | Polygon (list of coordinates) | **RLE** (Run-Length Encoding) |
| **Category IDs** | 1–91 (COCO 91 lớp) | **1–60** (60 lớp sản phẩm siêu thị) |
| **Image size** | Variable (COCO images) | 1440×1920 hoặc 1534×1920 (có zero-padding) |

### Splits D2SA

| Split | File | Images | Annotations |
|-------|------|--------|-------------|
| **Training (base)** | `D2S_amodal_training_rot0.json` | 438 | 690 |
| **Training (augmented)** | `D2S_amodal_augmented.json` | 1,562 | 12,376 |
| **Validation** | `D2S_amodal_validation.json` | 3,600 | 15,654 |
| **Test** | `D2S_amodal_test_info.json` | 13,020 | 0 (không có annotation!) |

> **Lưu ý:** Paper gốc D2SA dùng `training_rot0 + augmented` làm training set (tổng: 2,000 images, 13,066 annotations). Test set **không có annotation công khai** → **dùng validation set làm test set** khi evaluate.

### Sample annotation (object bị che — `occlude_rate > 0`):
```json
{
  "segmentation": {"counts": "...", "size": [1534, 1920]},    // Amodal mask (RLE)
  "area": 659199.0,
  "occl_depth": 0,
  "iscrowd": false,
  "amodal_region": {"name": "caona_...", "area": 659199.0, "isStuff": 0, "bbox": [...], "order": 1},
  "visible_mask": {"counts": "...", "size": [1534, 1920]},    // Visible mask (RLE)
  "invisible_mask": {"counts": "...", "size": [1534, 1920]},  // Invisible mask (RLE, optional)
  "image_id": 5300,
  "bbox": [228.0, 0.0, 1128.0, 1201.0],
  "occlude_rate": 0.019906,
  "category_id": 19,
  "id": 115,
  "amodal_base": {"url": "...", "image_id": 5300, "size": 1, "author": "MVTec Software GmbH", "depth_constraint": ""}
}
```

### Implications cho `dataset_d2sa.py`:
1. **Không cần tính visible mask** từ `order` như COCOA — D2SA cung cấp `visible_mask` trực tiếp (RLE).
2. **Amodal mask = `segmentation`** (RLE), visible mask = `visible_mask` (RLE) → decode bằng `pycocotools.mask.decode()`.
3. **Occluded region = amodal – visible** hoặc trực tiếp từ `invisible_mask` (nếu có).
4. **Edge mask** tính từ `visible_mask` (dilate – erode, giống COCOA).
5. **`num_classes=60`** — ID liên tục 1–60.

---

## 🐛 KNOWN ISSUES — Không sửa trong scope hiện tại

> Các vấn đề dưới đây được phát hiện trong quá trình đọc hiểu code ở Phase 1. **Không sửa** trong scope mở rộng D2SA để tránh rủi ro ảnh hưởng pipeline COCOA/KINS đang chạy tốt. Xử lý riêng sau khi D2SA hoàn tất.

### 1. Bug import sai ở Row 2 COCOA

**File:** `scripts/other_config_cocoa/amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce/train.py` dòng 28.

```python
# Hiện tại (có thể sai):
from scripts.model import AmodalSwinUNet

# Các config khác (Row 1, 3, 5, 6) đều dùng:
from model import AmodalSwinUNet
```

**Phân tích:** Config này import `AmodalSwinUNet` từ `scripts/model.py` (model gốc Row 4, **có** Category Embedding + 5 kênh), thay vì import từ `model.py` cục bộ trong cùng thư mục (model **không có** Embedding + chỉ 5 kênh). Nếu chạy từ working directory là thư mục config, lệnh `from scripts.model` có thể resolve sai hoặc đúng tuỳ cách `sys.path` được thiết lập. Cần kiểm tra lại xem checkpoint Row 2 COCOA đã train được tạo bằng model nào thật sự.

### 2. Naming convention không nhất quán trong `other_config_cocoa/`

- Row 1-3 dùng dấu gạch ngang: `amodal-shape-prediction-...`
- Row 5-6 dùng dấu gạch dưới: `amodal_shape_prediction_...`
- Typo "embeding" (thiếu `d`, đúng phải là "embedding") xuất hiện ở 4/5 thư mục
- Row 1-2 mô tả input+loss chi tiết (`rgb-vis-edge-bce`), Row 3/5/6 không

**Quyết định:** D2SA sẽ giữ nguyên tên y hệt COCOA (kể cả typo và inconsistency) để đảm bảo đối chiếu 1:1 an toàn khi viết báo cáo. Rename thống nhất (nếu cần) sẽ làm riêng sau khi D2SA hoàn tất, tách biệt khỏi scope thêm dataset.
