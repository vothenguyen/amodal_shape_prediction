# 🎯 PROMPT CHO DEVELOPER AGENT: TRIỂN KHAI ROW 7 — SEMANTIC-GUIDED GATED BOTTLENECK ATTENTION (SG-GBA) TRÊN D2SA, COCOA & KINS

## 1. MỤC TIÊU & BỐI CẢNH (CONTEXT)
- **Nhiệm vụ:** Mở rộng nghiên cứu Ablation Study từ 6 hàng (Row 1 -> Row 6) lên **Row 7 (Proposed Improved Architecture cho Khóa luận tốt nghiệp)**: Tích hợp khối **Semantic-Guided Gated Bottleneck Attention (`SGGatedBottleneck`)** ngay tại tầng Bottleneck (7x7x768) của `AmodalSwinUNet`.
- **Ý nghĩa khoa học:**
  1. Thay thế phép cộng cứng nhắc `x_bottleneck + c_emb` của Row 4 bằng cơ chế điều biến kênh **FiLM (Feature-wise Linear Modulation)** để khắc phục hiện tượng giảm Invisible mIoU trên các vật thể biến dạng phức tạp (COCOA).
  2. Chuyển cơ chế **Spatial Attention** từ cuối Decoder (gây ra *Spatial Attention Paradox*) về ngay tại **Bottleneck (7x7)** với kernel 3x3 và cổng **Residual Gate (alpha)** khởi tạo bằng `0.01` để nắm bắt ngữ cảnh đối xứng toàn cục trước khi giải mã mà không bị nghẽn đạo hàm (dead gradient) ở step đầu tiên.
- **Nguyên tắc kỹ thuật:**
  - Bám sát tuyệt đối `PROJECT_SUPERVISOR_HANDOFF.md`.
  - Tái sử dụng 100% `scripts/dataset_d2sa.py`, `scripts/dataset.py`, và `scripts/logging_utils.py`.
  - **KHÔNG** sửa đổi làm hỏng code của Row 1 -> Row 6 hiện có. Tạo thư mục cấu hình mới cho Row 7 theo đúng quy chuẩn đặt tên trong `scripts/other_config_d2sa/` và `scripts/other_config_cocoa/`, đồng thời cập nhật `scripts/run_ablation_d2sa.py` và `scripts/run_ablation_d2sa.sh` để hỗ trợ chạy Row 7.

---

## 2. YÊU CẦU KỸ THUẬT CHI TIẾT (SPECIFICATIONS)

### 2.1. Thiết kế Module `SGGatedBottleneck`
Đặt tại tầng Bottleneck (`[B, 768, 7, 7]`), nhận vào `x_bottleneck` (`[B, 768, 7, 7]`) và vector nhúng nhãn `c_emb` (`[B, 768]` từ `self.category_emb(class_ids)`):

```python
class SGGatedBottleneck(nn.Module):
    """
    Semantic-Guided Gated Bottleneck Attention (SG-GBA - Row 7).
    Kết hợp điều biến đặc trưng theo ngữ nghĩa (FiLM) và chú ý không gian toàn cục
    tại độ phân giải 7x7 với cổng nối tắt học được (Learnable Residual Gate).
    """
    def __init__(self, channels=768, reduction=4, spatial_kernel=3):
        super().__init__()
        hidden_dim = channels // reduction  # 768 // 4 = 192
        self.film_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2, bias=True),
        )
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        # Khởi tạo alpha = 0.01 để đảm bảo cả alpha, film_mlp và spatial_conv
        # đều nhận được gradient khác 0 ngay từ batch đầu tiên
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.last_attn_map = None

        # Zero-init lớp Linear cuối của FiLM để tại step đầu f_mod == x_bottleneck
        nn.init.zeros_(self.film_mlp[2].weight)
        nn.init.zeros_(self.film_mlp[2].bias)

    def forward(self, x_bottleneck, c_emb):
        # x_bottleneck: [B, 768, 7, 7], c_emb: [B, 768]
        film_params = self.film_mlp(c_emb).unsqueeze(-1).unsqueeze(-1)  # [B, 1536, 1, 1]
        gamma, beta = torch.chunk(film_params, 2, dim=1)               # Mỗi tensor [B, 768, 1, 1]

        # 1. Điều biến kênh theo ngữ nghĩa (FiLM)
        f_mod = (1.0 + torch.tanh(gamma)) * x_bottleneck + beta

        # 2. Chú ý không gian toàn cục tại Bottleneck (7x7)
        avg_out = torch.mean(f_mod, dim=1, keepdim=True)               # [B, 1, 7, 7]
        max_out, _ = torch.max(f_mod, dim=1, keepdim=True)             # [B, 1, 7, 7]
        attn_map = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )  # [B, 1, 7, 7]

        # Lưu lại attention map phục vụ trực quan hóa Heatmap trên Gradio UI (app.py)
        self.last_attn_map = attn_map.detach()

        # 3. Kết nối tắt có cổng học được (Learnable Residual Gate)
        out = x_bottleneck + self.alpha * (f_mod * attn_map)
        return out
```

### 2.2. Ràng buộc Số lượng Tham số (`EXPECTED_PARAMS` Assertion)
- So với Row 4 (`scripts/train_d2sa.py`), Row 7 giữ nguyên `self.category_emb = nn.Embedding(num_classes, 768)` và thêm khối `self.sg_bottleneck = SGGatedBottleneck(channels=768, reduction=4, spatial_kernel=3)`, **KHÔNG** dùng Spatial Attention ở cuối Decoder.
- Độ lệch tham số chính xác:
  - `film_mlp`: `(768 * 192 + 192) + (192 * 1536 + 1536) = 444,096`
  - `spatial_conv`: `2 * 1 * 3 * 3 = 18`
  - `alpha`: `1`
  - Tổng cộng `Delta = +444,115` tham số so với Row 4.
- **Trên D2SA (`num_classes=60`):**
  `EXPECTED_PARAMS = 34_399_643 + 444_115 = 34_843_758`
- **Trên COCOA (`num_classes=91`):**
  `EXPECTED_PARAMS = 34_423_451 + 444_115 = 34_867_566`
- Bắt buộc đặt lệnh `assert total_params == EXPECTED_PARAMS` khi khởi tạo model trong script train và evaluate của Row 7.

---

## 3. CÁC BƯỚC THỰC HIỆN TUẦN TỰ (PHASED TASKS)

### Bước 1: Kiểm tra cấu trúc hiện tại
- Đọc nhanh `scripts/model.py`, `scripts/train_d2sa.py`, `scripts/evaluate_d2sa.py`, `scripts/run_ablation_d2sa.py`, và 1 thư mục mẫu trong `scripts/other_config_d2sa/` (ví dụ `amodal_shape_prediction_full_config`) để nắm chuẩn xác cách tổ chức `model.py`, `train_d2sa.py`, `evaluate_d2sa.py`, đường dẫn lưu checkpoint và log.

### Bước 2: Tạo Cấu hình Row 7 cho D2SA và COCOA
1. Tạo thư mục `scripts/other_config_d2sa/amodal_shape_prediction_sg_gba_bottleneck/` gồm:
   - `model.py`: Chứa `AmodalSwinUNet` (5 kênh đầu vào, `num_classes=60`, `category_emb`, và `SGGatedBottleneck` ở bottleneck). Trong `forward(self, x, class_ids)`, thay dòng cộng trực tiếp `x_bottleneck = x_bottleneck + c_emb` bằng `c_emb = self.category_emb(class_ids)` rồi gọi `x_bottleneck = self.sg_bottleneck(x_bottleneck, c_emb)`.
   - `train_d2sa.py`: Kế thừa toàn bộ logic từ `scripts/train_d2sa.py` (OcclusionAwareLoss 5x + Dice, AdamW 1e-4, CosineAnnealingLR 30 epochs, Dual-level validation, `logging_utils`, atomic checkpoint, SIGTERM handler). Đặt `EXPECTED_PARAMS = 34_843_758`, thư mục lưu checkpoint mặc định là `checkpoints/d2sa/row7_sg_gba_bottleneck`, log prefix tương ứng, và ghi thêm giá trị `float(model.sg_bottleneck.alpha.item())` vào console/log sau mỗi epoch để theo dõi sự tiến hóa của cổng `alpha`.
   - `evaluate_d2sa.py`: Tương tự `scripts/evaluate_d2sa.py` nhưng import `AmodalSwinUNet` của Row 7 và kiểm tra `EXPECTED_PARAMS = 34_843_758`.
2. Tạo thư mục tương ứng cho COCOA tại `scripts/other_config_cocoa/amodal_shape_prediction_sg_gba_bottleneck/` (với `num_classes=91`, `EXPECTED_PARAMS = 34_867_566`) gồm `model.py`, `train.py`, `evaluate.py` theo chuẩn của `scripts/other_config_cocoa/`.
3. Cập nhật `scripts/run_ablation_d2sa.py` và `scripts/run_ablation_d2sa.sh`:
   - Bổ sung cấu hình **Row 7 (`row7_sg_gba_bottleneck`)** vào danh sách mapping của runner để có thể gọi `--rows 7` hoặc `--rows 1,2,3,4,5,6,7`.

### Bước 3: Chạy Smoke Test trên máy Local (4GB VRAM)
- Thực thi Smoke Test cho Row 7 trên D2SA (sử dụng batch size nhỏ = 2):
  1. Khởi tạo model Row 7 (D2SA `num_classes=60` và COCOA `num_classes=91`), kiểm tra `assert total_params == 34_843_758` (D2SA) và `34_867_566` (COCOA).
  2. Chạy thử 2 step forward + backward + optimizer step liên tiếp trên GPU local.
  3. Xác nhận `model.sg_bottleneck.alpha.grad`, `model.sg_bottleneck.film_mlp[0].weight.grad`, `model.sg_bottleneck.film_mlp[2].weight.grad`, và `model.sg_bottleneck.spatial_conv.weight.grad` đều khác `None`, khác `0`, và không có `NaN`.
  4. Kiểm tra `model.sg_bottleneck.last_attn_map.shape == (2, 1, 7, 7)`.
  5. Chạy thử `python scripts/run_ablation_d2sa.py --rows 7 --epochs 1` (hoặc chế độ smoke test tương ứng của repo) và kiểm tra lưu/đọc `last.pth` thành công.

---

## 4. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)
- [ ] Khớp chính xác `EXPECTED_PARAMS = 34_843_758` (D2SA) và `34_867_566` (COCOA).
- [ ] Qua bài test gradient: chứng minh được cả `alpha`, `film_mlp` và `spatial_conv` đều nhận được gradient hợp lệ khác 0.
- [ ] `scripts/run_ablation_d2sa.py --rows 7` hoạt động trơn tru.
- [ ] Không làm thay đổi bất kỳ logic nào của Row 1 -> Row 6.
