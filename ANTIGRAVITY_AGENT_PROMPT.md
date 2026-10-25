# PROMPT CHO AGENT (Antigravity) — Dự án Amodal Shape Prediction — Mở rộng sang D2SA

> Cách dùng: dán toàn bộ nội dung dưới đây vào khung chat của agent trong Antigravity, trong đúng thư mục gốc của repo `amodal_shape_prediction` đã giải nén. Agent PHẢI làm theo đúng thứ tự các Phase, không được nhảy cóc, không được tự ý push/commit lên GitHub hoặc thêm collaborator.

---

## 0. BỐI CẢNH DỰ ÁN (đọc kỹ trước khi làm bất cứ điều gì)

Đây là đồ án/khóa luận tốt nghiệp về **Amodal Shape Prediction** (dự đoán hình dạng đầy đủ của vật thể bị che khuất), pipeline 2 giai đoạn:

- **Stage 1:** SAM 2.1 trích Visible Mask từ point prompt (zero-shot, không train).
- **Stage 2 (phần chính cần train):** `Swin-UNet` 5-channel (RGB + Visible Mask + Edge Mask) + Category Embedding (768-dim, cộng vào bottleneck) → dự đoán Amodal Mask `[B,1,224,224]`.

Cấu hình chính (Row 4 trong Table V - Ablation Study) là: **Occlusion-Aware Loss + Edge Mask + Category Embedding = True, Spatial Attention = False**. Dự án đã hỗ trợ **6 cấu hình ablation** (Row 1–6) khác nhau về Loss / Edge Mask / Category Embedding / Spatial Attention.

Dự án **đã train xong và có kết quả trên 2 dataset**: **COCOA** (script `scripts/train.py` + các biến thể ablation ở `scripts/other_config_cocoa/`) và **KINS** (`scripts/train_kins.ipynb`). Mỗi dataset đã chạy đủ **6 cấu hình ablation x 30 epochs**.

**Mục tiêu lần này:** mở rộng sang dataset thứ 3 là **D2SA (D2S Amodal)** — đây cũng là bộ ba dataset chuẩn trong literature về amodal segmentation (COCOA, KINS, D2SA thường đi cùng nhau, ví dụ trong paper VRSP-Net, AISFormer). Chỉ code + kiểm thử (smoke test) ở máy local, **không** train full ở đây (train full 6x30 epochs sẽ do giảng viên chạy trên A100 của trường sau khi merge code).

---

## PHASE 1 — ĐỌC HIỂU TOÀN BỘ CONTEXT (bắt buộc, không được bỏ qua)

Nhiệm vụ: đọc và tóm tắt lại (bằng tiếng Việt, viết ra file `docs/PROJECT_UNDERSTANDING.md`) sự hiểu biết của agent về:

1. `README.md` gốc — kiến trúc, format dữ liệu, cấu trúc thư mục.
2. `scripts/README.md` — chi tiết 6 cấu hình ablation (Row 1–6) khác nhau chỗ nào (loss, edge mask, category embedding, spatial attention).
3. `scripts/model.py` — kiến trúc Swin-UNet, cách category embedding được cộng vào, input/output shape, **số lớp (num_classes) hiện đang hard-code hay tham số hoá**.
4. `scripts/dataset.py` — logic đọc annotation COCOA-format (`regions`, `segmentation`, `order`), cách sinh Visible Mask từ `order`, cách sinh Edge Mask (cv2.dilate/erode), augmentation bằng `albumentations`.
5. `scripts/train.py` — vòng lặp train, optimizer, scheduler, loss (Occlusion-Aware Loss), cách lưu checkpoint, cách log.
6. `scripts/evaluate.py` — cách tính mIoU, Dice, Precision, Recall (single-image), format output JSON/PNG ở `results/`.
7. `scripts/train_kins.ipynb` — xem KINS được thích nghi (adapt) từ pipeline COCOA như thế nào: annotation format của KINS khác COCOA ở đâu, dataset class có viết lại hay tái sử dụng, num_classes có đổi không.
8. `scripts/other_config_cocoa/` — cách tổ chức code cho 6 biến thể ablation (file riêng từng config, hay 1 file + flag/config?). **Đây là điểm quan trọng nhất để quyết định cách tổ chức code cho D2SA ở Phase 3**, vì yêu cầu là làm **theo đúng pattern đã có**, không tự sáng tạo cấu trúc mới.
9. `app.py`, `requirements.txt`, `.gitignore`, `temp.py` — xác định file nào là rác/thử nghiệm dở dang (đặc biệt `temp.py` — nếu là file scratch không dùng, đưa vào đề xuất dọn dẹp ở Phase 2, KHÔNG tự xoá khi chưa hỏi).
10. Checkpoints hiện có trong `checkpoints/` (nếu tồn tại trong bản giải nén) — SAM 2.1 weight, Swin checkpoint COCOA/KINS.

Sau khi đọc xong, agent phải in ra bản tóm tắt ngắn gọn (checklist) và **hỏi lại tớ (người dùng) nếu có gì mâu thuẫn với mô tả ở prompt này** trước khi sang Phase 2.

---

## PHASE 2 — ĐỀ XUẤT GOM GỌN / SẮP XẾP FILE (không bắt buộc phải làm, chỉ đề xuất)

- Nêu ra danh sách file/thư mục thừa, trùng lặp, đặt tên không nhất quán, hoặc là rác (candidate: `temp.py` ở root).
- Đề xuất vị trí đặt các file mới cho D2SA sao cho **nhất quán với pattern COCOA/KINS đã có** (xem Phase 1, mục 8) — ví dụ nếu COCOA dùng `other_config_cocoa/` cho 6 biến thể thì D2SA nên có `other_config_d2sa/` tương ứng.
- Chỉ thực hiện xoá/di chuyển file sau khi tớ xác nhận đồng ý với đề xuất. Không tự động xoá bất cứ thứ gì có khả năng ảnh hưởng tới kết quả COCOA/KINS đã có sẵn.

---

## PHASE 3 — CHUẨN BỊ DỮ LIỆU D2SA

### 3.1. Tải dữ liệu (agent tự tải giúp, không cần tớ tải tay)

D2SA (D2S Amodal) là annotation amodal bổ sung cho MVTec D2S (Densely Segmented Supermarket, 60 lớp sản phẩm siêu thị). Link tải trực tiếp (không cần đăng nhập/tài khoản), lấy từ trang chính thức MVTec (`https://www.mvtec.com/research-teaching/datasets/mvtec-d2s/downloads`):

- **Ảnh D2S Amodal (4.2 GB):**
  `https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420939124-1629954676/d2s_amodal_images_v1.tar.xz`
- **Annotation Amodal (15.3 MB):**
  `https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938643-1629954673/d2s_amodal_annotations_v1.tar.xz`

Agent hãy tự `wget`/`curl` 2 file này, giải nén vào `data/D2SA/` theo đúng convention thư mục `data/` đã dùng cho COCOA/KINS (xem Phase 1). Nếu link nào lỗi/hết hạn, agent báo lại cho tớ thay vì tự đoán URL khác.

> ⚠️ **Lưu ý quan trọng — PHẢI xử lý đúng:** Tập **test** của D2S/D2SA **không có annotation công khai** (MVTec giữ private để họ tự chấm điểm nếu mình gửi kết quả cho họ). Vì vậy pipeline D2SA của mình phải dùng: **train split để train**, **validation split để vừa validate vừa đóng vai trò "test" báo cáo trong khoá luận** (ghi rõ điều này trong README/báo cáo để không bị hiểu nhầm là test set thật). Đây là cách các paper khác (VRSP-Net, AISFormer) cũng làm.

### 3.2. Kiểm tra format annotation

Trước khi viết `dataset_d2sa.py` (hay tên tương ứng theo pattern đã chọn ở Phase 2), agent phải:
- Mở thử file JSON annotation D2SA, in ra cấu trúc field thực tế (không đoán).
- So sánh với format COCOA hiện tại (`regions`, `segmentation`, `order` dùng để suy ra Visible Mask).
- Nếu D2SA đã có sẵn cả amodal mask lẫn visible mask trong annotation (khác với COCOA phải suy `order`), agent phải điều chỉnh logic dataset cho đúng — **không được ép format D2SA giống COCOA nếu thực tế khác**. Ghi rõ khác biệt này vào `docs/PROJECT_UNDERSTANDING.md`.
- Nếu ảnh D2SA có kích thước/tỷ lệ khác nhiều so với COCO (ảnh siêu thị chụp cận), kiểm tra xem bước resize `224x224` bằng `albumentations` có cần augmentation riêng không (không bắt buộc đổi, nhưng phải kiểm tra).

### 3.3. Category Embedding cho D2SA

D2S có khoảng **60 lớp sản phẩm**, hoàn toàn khác 90 lớp COCO. Quyết định: **train một Category Embedding riêng cho D2SA** theo đúng số lớp thực tế của D2S (không tái sử dụng embedding table của COCOA/KINS). Kiến trúc Swin-UNet giữ nguyên, chỉ tham số hoá `num_classes` theo dataset đang train (nếu code hiện tại hard-code 90, cần sửa thành tham số/config).

---

## PHASE 4 — TỔ CHỨC CODE TRAIN CHO D2SA

- Bám sát đúng pattern đã xác định ở Phase 1/2 (khả năng cao là mirror theo cấu trúc COCOA vì cần chạy hệ thống 6 cấu hình ablation, không phải 1 notebook đơn lẻ như KINS — nhưng **agent phải tự kiểm tra lại cấu trúc `other_config_cocoa/` thực tế rồi mới quyết định**, không suy đoán).
- Tạo bộ 6 file/cấu hình ablation cho D2SA tương ứng **chính xác Row 1–6** trong Table V (loss, edge mask, category embedding, spatial attention) — copy logic từ COCOA, chỉ đổi phần dataset/category embedding/num_classes.
- **Giữ nguyên toàn bộ hyperparameter (learning rate, optimizer, scheduler, batch size gốc, số epoch = 30/config)** giống hệt COCOA/KINS để đảm bảo so sánh công bằng giữa 3 dataset.
- Đảm bảo `evaluate.py` chạy được với checkpoint D2SA, xuất đúng format JSON/PNG (mIoU, Dice, Precision, Recall) giống COCOA/KINS để dễ tổng hợp bảng so sánh 3 dataset trong khoá luận.

---

## PHASE 5 — SMOKE TEST TRÊN MÁY LOCAL (RTX 3050 4GB VRAM)

**Không train full ở đây.** Máy local chỉ có 4GB VRAM, mục tiêu là xác nhận pipeline chạy đúng logic (không lỗi shape, không lỗi loss NaN, checkpoint lưu/load được, evaluate chạy được), không phải để ra kết quả cuối.

Yêu cầu agent:
1. Tạo một subset rất nhỏ của D2SA (ví dụ 30–50 ảnh train, 10–20 ảnh val) để test nhanh, không cần load hết 4.2GB ảnh.
2. Giảm batch size (ví dụ 2–4), bật mixed precision (`torch.cuda.amp`) nếu code chưa có, để vừa VRAM 4GB. Nếu vẫn tràn VRAM, cho phép chạy CPU cho phần smoke test (chấp nhận chậm) — chỉ cần chứng minh code chạy không lỗi, không cần tốc độ.
3. Chạy thử **cả 6 cấu hình ablation, mỗi cấu hình 1–2 epoch** trên subset nhỏ này để đảm bảo cả 6 file config đều chạy được, không riêng cấu hình chính (Row 4).
4. Chạy thử `evaluate.py` trên checkpoint vừa tạo ra từ bước smoke test, xác nhận metric tính ra hợp lý (không NaN, không lỗi format).
5. Báo cáo lại kết quả smoke test (log ngắn gọn: config nào pass, config nào lỗi và vì sao) trước khi coi Phase 5 là hoàn thành.

---

## PHASE 6 — CHUẨN BỊ CHO GIẢNG VIÊN TRAIN FULL TRÊN A100

Vì COCOA và KINS đã có kết quả đầy đủ rồi, phần này chỉ cần chuẩn bị cho **D2SA, đủ 6 cấu hình x 30 epochs**:

1. Viết 1 script chạy tuần tự cả 6 cấu hình (ví dụ `scripts/run_ablation_d2sa.sh` hoặc notebook tương đương theo đúng pattern đã chọn), có tham số chỉnh device/GPU, checkpoint output path riêng cho từng config.
2. Cập nhật `requirements.txt` nếu cần thêm thư viện nào cho D2SA (không đổi các thư viện đã dùng cho COCOA/KINS nếu không cần thiết).
3. Viết hướng dẫn ngắn (README hoặc phần thêm vào `scripts/README.md`) mô tả: cách tải dữ liệu D2SA đầy đủ trên server, cách chạy 6 cấu hình, thời gian ước tính, nơi kết quả sẽ được lưu (`results/`, `checkpoints/`) để giảng viên chỉ cần `git pull` + chạy script.
4. Ghi rõ trong README lưu ý về việc D2SA dùng validation set thay test set thật (do MVTec không công khai test annotation).

---

## PHASE 7 — GIT (chỉ chuẩn bị, KHÔNG tự thực hiện)

- Agent **KHÔNG được** tự chạy `git push`, tạo remote, hoặc thêm collaborator trên GitHub dưới bất kỳ hình thức nào.
- Agent chỉ nên: đảm bảo code sạch, có thể gợi ý commit message hợp lý theo từng phase, và (nếu tớ yêu cầu) tạo commit **local** để tớ tự kiểm tra trước khi `git push` bằng tay.
- Việc thêm giảng viên làm collaborator trên GitHub, và việc push code lên remote, tớ sẽ tự làm sau khi review toàn bộ code.

---

## QUY TẮC LÀM VIỆC CHUNG CHO AGENT

1. Luôn đọc code thật (không suy đoán) trước khi viết code mới — đặc biệt là format annotation D2SA thật và cấu trúc `other_config_cocoa/` thật.
2. Trước khi thực hiện một Phase mới, tóm tắt ngắn gọn kế hoạch cụ thể của Phase đó và hỏi xác nhận nếu có quyết định ảnh hưởng tới cấu trúc code hiện có.
3. Không sửa/xoá bất cứ thứ gì liên quan đến pipeline COCOA hoặc KINS đã chạy xong, trừ khi được yêu cầu rõ ràng.
4. Giữ nguyên toàn bộ hyperparameter huấn luyện (để so sánh 3 dataset công bằng) — chỉ được thay đổi phần liên quan tới dataset-specific (num_classes, category embedding, augmentation nếu bắt buộc).
5. Trả lời và ghi chú code bằng tiếng Việt là ưu tiên (comment tiếng Anh trong code là bình thường, nhưng giải trình/README nên có tiếng Việt vì phục vụ khoá luận).
6. Nếu phát hiện annotation D2SA có gì bất thường (thiếu field, format khác dự kiến, ảnh lỗi), dừng lại và báo cáo thay vì tự "vá" logic để chạy được.
