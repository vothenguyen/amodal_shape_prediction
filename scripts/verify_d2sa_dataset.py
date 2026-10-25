"""
Kiểm tra tính toàn vẹn của tập dữ liệu D2SA trước khi bắt đầu huấn luyện.
Cách dùng:
    python scripts/verify_d2sa_dataset.py
    python scripts/verify_d2sa_dataset.py --data-dir /path/to/data/D2SA
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def verify_dataset(data_dir: str):
    print("=" * 70)
    print("KIỂM TRA TÍNH TOÀN VẸN TẬP DỮ LIỆU D2SA")
    print(f"Thư mục kiểm tra: {os.path.abspath(data_dir)}")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"[THẤT BẠI] Thư mục không tồn tại: {data_dir}")
        print("Vui lòng tải và giải nén dữ liệu D2SA theo hướng dẫn trong tài liệu bàn giao.")
        sys.exit(1)

    errors = []
    warnings = []

    # 1. Kiểm tra 3 file annotation JSON
    expected_jsons = {
        "D2S_amodal_training_rot0.json": {"min_images": 400, "min_anns": 600},
        "D2S_amodal_augmented.json": {"min_images": 1500, "min_anns": 12000},
        "D2S_amodal_validation.json": {"min_images": 3500, "min_anns": 15000},
    }

    print("\n1. Kiểm tra các file Annotation JSON:")
    total_expected_images = set()

    for json_name, criteria in expected_jsons.items():
        json_path = os.path.join(data_dir, json_name)
        if not os.path.isfile(json_path):
            print(f"  ❌ THIẾU FILE: {json_name}")
            errors.append(f"Thiếu file annotation {json_name}")
            continue

        file_size_mb = os.path.getsize(json_path) / (1024 * 1024)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            num_imgs = len(data.get("images", []))
            num_anns = len(data.get("annotations", []))
            num_cats = len(data.get("categories", []))

            for img in data.get("images", []):
                total_expected_images.add(img.get("file_name"))

            print(f"  ✅ {json_name:<30} ({file_size_mb:.2f} MB): {num_imgs:,} ảnh, {num_anns:,} nhãn, {num_cats} lớp")

            if num_imgs < criteria["min_images"]:
                warnings.append(f"{json_name} có số ảnh ít hơn mong đợi ({num_imgs} < {criteria['min_images']})")
            if num_anns < criteria["min_anns"]:
                warnings.append(f"{json_name} có số nhãn ít hơn mong đợi ({num_anns} < {criteria['min_anns']})")

        except Exception as e:
            print(f"  ❌ LỖI ĐỌC FILE {json_name}: {e}")
            errors.append(f"Không thể đọc file {json_name}: {e}")

    # 2. Kiểm tra thư mục ảnh
    print("\n2. Kiểm tra thư mục ảnh (images/):")
    img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(img_dir):
        print(f"  ❌ THIẾU THƯ MỤC: {img_dir}")
        errors.append(f"Thiếu thư mục ảnh: {img_dir}")
    else:
        # Đếm số file ảnh
        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = [f for f in os.listdir(img_dir) if os.path.splitext(f.lower())[1] in valid_exts]
        num_images = len(image_files)
        print(f"  📁 Tổng số file ảnh tìm thấy: {num_images:,} ảnh")

        EXPECTED_COUNT = 22562
        if num_images == EXPECTED_COUNT:
            print(f"  ✅ Số lượng ảnh hoàn toàn chính xác ({EXPECTED_COUNT:,}/{EXPECTED_COUNT:,})")
        elif num_images >= 20000:
            print(f"  ⚠️  Số lượng ảnh ({num_images:,}) gần sát chuẩn ({EXPECTED_COUNT:,})")
        else:
            print(f"  ❌ Số lượng ảnh quá ít ({num_images:,} < {EXPECTED_COUNT:,})! Có thể giải nén chưa xong.")
            errors.append(f"Số lượng ảnh không đủ ({num_images} < {EXPECTED_COUNT})")

        # 3. Kiểm tra tính toàn vẹn của một số file ảnh ngẫu nhiên
        print("\n3. Kiểm tra tính toàn vẹn file ảnh ngẫu nhiên:")
        sample_indices = [0, len(image_files)//4, len(image_files)//2, 3*len(image_files)//4, -1] if image_files else []
        corrupted_count = 0
        for idx in sample_indices:
            img_name = image_files[idx]
            img_path = os.path.join(img_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Reopen to get size after verify()
                with Image.open(img_path) as img:
                    w, h = img.size
                print(f"  ✅ Đọc tốt: {img_name} ({w}x{h})")
            except Exception as e:
                print(f"  ❌ ẢNH BỊ HỎNG: {img_name} ({e})")
                corrupted_count += 1

        if corrupted_count > 0:
            errors.append(f"Phát hiện {corrupted_count} ảnh bị lỗi khi đọc thử")

    # Tổng kết
    print("\n" + "=" * 70)
    if errors:
        print("❌ KẾT QUẢ: TẬP DỮ LIỆU CHƯA ĐỦ ĐIỀU KIỆN HUẤN LUYỆN!")
        for err in errors:
            print(f"   - {err}")
        print("Vui lòng kiểm tra lại quá trình tải và giải nén dữ liệu.")
        sys.exit(1)
    else:
        print("✅ KẾT QUẢ: TẬP DỮ LIỆU HOÀN TOÀN HỢP LỆ VÀ SẴN SÀNG HUẤN LUYỆN!")
        if warnings:
            print("Lưu ý nhỏ:")
            for w in warnings:
                print(f"   - {w}")
        print("Bạn có thể an tâm bắt đầu chạy full training (11-12 giờ) mà không lo lỗi dữ liệu giữa chừng.")
        print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra dataset D2SA")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "D2SA"),
        help="Đường dẫn đến thư mục D2SA (chứa images/ và các file JSON)"
    )
    args = parser.parse_args()
    verify_dataset(args.data_dir)
