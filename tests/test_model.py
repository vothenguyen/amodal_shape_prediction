"""
===================================================================================
TEST MÔ HÌNH NHANH - Demo với dữ liệu từ training set
===================================================================================
Script test đơn giản để kiểm tra mô hình hoạt động hay không.

Tính năng:
- Tải 1 mẫu ngẫu nhiên từ training set
- Chạy qua mô hình
- Hiển thị kết quả: Ảnh gốc, Visible mask, Dự đoán, Đáp án

Chạy: python src/test_model.py
===================================================================================
"""

import torch
import random
import matplotlib.pyplot as plt
import albumentations as A

# Import từ project
from model import AmodalSwinUNet
from dataset import AmodalDataset


def test_model():
    """
    Hàm test đơn giản.
    """
    # ─────────────────────────────────────────────────────────────
    # BƯỚC 1: THIẾT LẬP THIẾT BỊ
    # ─────────────────────────────────────────────────────────────
    # Chạy trên CPU (máy nhà có thể không có GPU)
    DEVICE = torch.device("cpu")
    print("🔧 Khởi động mô hình trên CPU...")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 2: NẠP MÔ HÌNH
    # ─────────────────────────────────────────────────────────────
    model = AmodalSwinUNet().to(DEVICE)
    weight_path = "../checkpoints/swin_amodal_epoch_30.pth"

    # Nạp trọng số (map_location để tránh lỗi GPU)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()  # Bật chế độ đi thi (không training)
    print(f"✅ Đã nạp trọng số từ: {weight_path}")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 3: CHUẨN BỊ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────
    img_dir = "../data/train2014"
    ann_file = "../data/annotations/COCO_amodal_train2014.json"

    # Lưu ý: Test chỉ resize, không augment
    test_transform = A.Compose([A.Resize(224, 224)])
    dataset = AmodalDataset(
        img_dir=img_dir, ann_file=ann_file, transform=test_transform
    )
    print(f"📂 Dataset: {len(dataset)} mẫu")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 4: LẤY MẪU NGẪU NHIÊN
    # ─────────────────────────────────────────────────────────────
    idx = random.randint(0, len(dataset) - 1)
    input_tensor, target_mask, occluded, class_id = dataset[idx]
    print(f"🎲 Mẫu ngẫu nhiên: #{idx}, Class ID: {class_id}")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 5: CHUYỂN THÀNH BATCH & DỰ ĐOÁN
    # ─────────────────────────────────────────────────────────────
    # Thêm chiều batch: [5, 224, 224] → [1, 5, 224, 224]
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    class_id_batch = torch.tensor([class_id]).to(DEVICE)

    print(f"🧠 Chạy mô hình...")
    with torch.no_grad():
        output_logits = model(input_batch, class_id_batch)
        pred_mask = torch.sigmoid(output_logits)
        # Chuyển thành binary mask (0 hoặc 1)
        pred_mask = (pred_mask > 0.5).squeeze().numpy()

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 6: TÁCH KÊNH & HIỂN THỊ
    # ─────────────────────────────────────────────────────────────
    # Tách từng kênh từ input tensor
    img_rgb = input_tensor[:3].numpy().transpose(1, 2, 0)
    visible_mask = input_tensor[3].numpy()
    truth_mask = target_mask.numpy()

    # Vẽ kết quả
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Ô 1: Ảnh RGB gốc
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Ảnh RGB Gốc", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    # Ô 2: Visible mask (từ SAM hoặc annotation)
    axes[1].imshow(visible_mask, cmap="gray")
    axes[1].set_title("2. Visible Mask (Phần nhìn thấy)", fontsize=12, fontweight='bold')
    axes[1].axis("off")

    # Ô 3: Dự đoán từ Swin-UNet
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("3. Dự Đoán Amodal (Epoch 30)", fontsize=12, fontweight='bold', color='blue')
    axes[2].axis("off")

    # Ô 4: Đáp án đúng
    axes[3].imshow(truth_mask, cmap="gray")
    axes[3].set_title("4. Đáp Án Thực Tế", fontsize=12, fontweight='bold')
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_model()
