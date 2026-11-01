# PROMPT CHO AGENT — Xây dựng hệ thống Logging bền vững khi train trên server hàng đợi (queue) của trường

> Dán nguyên văn vào agent Antigravity. Mục tiêu: đảm bảo **không mất bất kỳ kết quả nào** khi job bị xoá màn hình/terminal lúc hết lượt trong hàng đợi (queue) của server trường, kể cả khi job bị kill giữa chừng không báo trước.

---

## BỐI CẢNH — TẠI SAO CẦN VIỆC NÀY

Server GPU của trường chạy theo cơ chế hàng đợi (queue): khi đến lượt job tiếp theo, output đang hiển thị trên terminal/log mặc định của job trước **bị xoá sạch, không truy xuất lại được**. Vì vậy:
- Không được phép chỉ dùng `print()` để xem kết quả — phải **ghi ra file thật trên đĩa**.
- File log phải được **ghi ngay lập tức (flush) sau mỗi dòng quan trọng**, không được giữ trong bộ nhớ rồi ghi 1 lần lúc kết thúc — vì nếu job bị kill giữa chừng (hết giờ, hết quota, lỗi phần cứng, v.v.) mà chưa kịp ghi, toàn bộ kết quả từ đầu tới lúc đó sẽ mất trắng.
- Việc này áp dụng cho **cả 6 cấu hình ablation của D2SA** (và nên áp dụng ngược lại cho scripts hiện có của COCOA/KINS nếu tiện, nhưng không bắt buộc sửa code cũ đang chạy tốt).

---

## YÊU CẦU CHI TIẾT

### 1. Vị trí lưu log

Tạo thư mục `logs/` ở gốc dự án (ngang hàng với `scripts/`, `checkpoints/`, `results/`), có cấu trúc:

```
logs/
└── d2sa/
    ├── row1_bce_vis_noedge_noemb_nospatial/
    │   ├── 20260115_143022_train.log        # log dạng text, đọc được bằng mắt
    │   └── 20260115_143022_metrics.jsonl     # log dạng JSON Lines, 1 dòng/epoch, dễ parse sau này
    ├── row2_.../
    ...
    └── row4_main_config/
```

Mỗi lần chạy train (dù là chạy lại/resume) tạo **tên file mới theo mốc thời gian lúc bắt đầu chạy**, không ghi đè lên log của lần chạy trước — để nếu lần trước bị lỗi/kill giữa chừng, vẫn còn log cũ để đối chiếu.

### 2. Đặt tên file theo mốc thời gian

Format bắt buộc: `{YYYYMMDD}_{HHMMSS}_{loại_file}.{ext}`, ví dụ:
- `20260115_143022_train.log`
- `20260115_143022_metrics.jsonl`

Lấy timestamp ngay lúc script bắt đầu chạy (`datetime.now().strftime("%Y%m%d_%H%M%S")`), dùng chung 1 timestamp cho cả 2 file log của cùng 1 lần chạy để dễ đối chiếu.

### 3. Hai loại file log, ghi song song

**a) File `.log` dạng text** (dành cho đọc bằng mắt, giống terminal output nhưng lưu vĩnh viễn):
- Dùng module `logging` chuẩn của Python, KHÔNG dùng `print()` cho bất cứ thông tin nào cần giữ lại.
- Cấu hình đồng thời 2 handler: `FileHandler` (ghi ra file) và `StreamHandler` (vẫn hiện ra terminal để theo dõi trực tiếp nếu đang ngồi xem) — để không mất khả năng debug real-time khi đang có mặt.
- **Bắt buộc ghi file ở chế độ line-buffered hoặc flush thủ công sau mỗi dòng quan trọng** (mỗi epoch, mỗi lần lưu checkpoint, mỗi lỗi/exception) — không được để mặc định buffer của hệ điều hành giữ dữ liệu trong RAM chưa ghi xuống đĩa.

**b) File `.jsonl` (JSON Lines) dạng máy đọc được:**
- Mỗi dòng là 1 object JSON hoàn chỉnh, ghi + flush ngay sau mỗi epoch (không đợi hết cả quá trình train mới ghi).
- Mục đích: sau này viết script tổng hợp kết quả 6 cấu hình × 3 dataset thành bảng so sánh cho báo cáo khoá luận, đọc JSON dễ và ít lỗi hơn parse text bằng regex.

### 4. Nội dung bắt buộc phải log

**Lúc bắt đầu chạy (ghi 1 lần, ngay đầu file):**
- Tên cấu hình ablation (row mấy, dataset nào)
- Toàn bộ hyperparameter: batch size, learning rate, số epoch dự kiến, optimizer, scheduler, occlusion_weight (nếu có), num_classes
- Đường dẫn data (`img_dir`, `ann_file`)
- Thông tin môi trường: tên GPU (`torch.cuda.get_device_name()`), phiên bản CUDA, phiên bản PyTorch
- Thời gian bắt đầu (timestamp đầy đủ, có cả ngày)
- Nếu có thể lấy được: mã băm commit git hiện tại (`git rev-parse HEAD`) — để biết chính xác phiên bản code nào tạo ra kết quả này, phòng khi sau này code có thay đổi mà không nhớ rõ.

**Mỗi epoch (ghi ngay sau khi epoch đó hoàn tất, KHÔNG gộp chờ cuối):**
- Số epoch hiện tại / tổng số epoch
- Train loss (và các thành phần loss nếu có tách riêng BCE/Dice)
- Thời gian chạy epoch đó (giây)
- Đường dẫn checkpoint vừa lưu (nếu epoch đó có lưu checkpoint)
- Nếu có chạy validation trong lúc train: mIoU, Occlusion mIoU, Precision, Recall

**Khi có lỗi/exception:**
- Bọc toàn bộ vòng lặp train chính trong `try/except Exception`, log đầy đủ traceback (`logging.exception(...)` hoặc `traceback.format_exc()`) vào file log **trước khi** chương trình dừng/crash — không được để exception làm crash chương trình mà chưa kịp ghi log.
- Ghi rõ epoch nào đang chạy dở khi lỗi xảy ra, để biết cần resume từ đâu.

**Lúc kết thúc (dù thành công hay bị dừng giữa chừng, dùng `try/finally`):**
- Thời gian kết thúc, tổng thời gian chạy
- Epoch cuối cùng hoàn thành
- Đường dẫn checkpoint cuối cùng
- Nếu chạy hết: kết quả evaluate cuối cùng (Overall mIoU, Occlusion mIoU, Precision, Recall)

### 5. Bắt buộc dùng `try/finally` bao quanh toàn bộ hàm train chính

Để đảm bảo dù chương trình bị kill bằng SIGTERM (thường xảy ra khi server hết giờ chạy cho phép, có cảnh báo trước vài giây) thì vẫn kịp ghi dòng cuối vào log trước khi thoát. Nếu framework hỗ trợ, đăng ký thêm signal handler cho `SIGTERM` để chủ động ghi log + lưu checkpoint khẩn cấp ngay khi nhận tín hiệu server sắp kill, thay vì chỉ trông chờ vào `finally`.

### 6. Áp dụng cho cả 6 cấu hình ablation

Vì mỗi cấu hình ablation là 1 bản sao độc lập (theo đúng pattern `other_config_cocoa/` đã thống nhất trước đó — mỗi thư mục có `train.py` riêng, không import chung), viết 1 module logging dùng chung dạng `logging_utils.py`, rồi **copy y hệt file này vào cả 6 thư mục cấu hình D2SA** (giống cách các file khác đã được copy độc lập), để giữ đúng tính độc lập giữa các cấu hình như quy ước đã có.

### 7. Không tự chạy full training

Nhắc lại phạm vi: agent chỉ viết và test hệ thống logging này bằng smoke test (1-2 epoch trên subset nhỏ, như đã làm ở Phase 5), xác nhận file `.log` và `.jsonl` được tạo ra đúng, có nội dung đầy đủ như yêu cầu ở mục 4, rồi báo cáo lại — **không tự ý chạy full training 30 epoch x 6 config trên server thật**, việc đó dành cho giảng viên chạy trên A100.

---

## OUTPUT MONG ĐỢI SAU KHI AGENT LÀM XONG

1. File `scripts/other_config_d2sa/logging_utils.py` (hoặc tên tương tự), được copy vào tất cả 6 thư mục cấu hình.
2. Từng `train.py` của 6 cấu hình được cập nhật để gọi module logging này, thay mọi `print()` quan trọng bằng `logging.info(...)`.
3. Kết quả smoke test: đường dẫn tới 1 file `.log` và 1 file `.jsonl` mẫu thực tế đã tạo ra, để tớ mở xem thử nội dung có đúng như mục 4 yêu cầu không.
4. Xác nhận đã test tình huống lỗi giả lập (ví dụ chủ động raise exception giữa chừng 1 epoch) để chắc chắn log vẫn ghi được traceback đầy đủ trước khi chương trình dừng.
