"""
===================================================================================
BATCH TEST (MODEL PRO) - Chạy batch ảnh & hiển thị lưới kết quả
===================================================================================
Script để visualize kết quả của mô hình trên batch ảnh.

Tính năng:
- Xử lý song song 4 ảnh (batch)
- Hiển thị kết quả dưới dạng lưới 2×2
- Hỗ trợ GPU acceleration

Chạy: python src/batch_test_pro.py
===================================================================================
"""

import os
import torch
import random
import matplotlib.pyplot as plt
import albumentations as A
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

# Import từ project
from model import AmodalSwinUNet
from dataset import AmodalDataset


def batch_test_pro():
    """
    Chạy test trên batch ảnh (4 ảnh cùng lúc).
    """
    # ─────────────────────────────────────────────────────────────
    # BƯỚC 1: KHỞI ĐỘNG
    # ─────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chạy trên thiết bị: {DEVICE}")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 2: NẠP MÔ HÌNH
    # ─────────────────────────────────────────────────────────────
    # Mô hình hỗ trợ 91 class COCO
    model = AmodalSwinUNet(num_classes=91).to(DEVICE)
    
    TEST_EPOCH = 30
    weight_path = f'../checkpoints/swin_amodal_epoch_{TEST_EPOCH}.pth'
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"✅ Đã nạp mô hình Epoch {TEST_EPOCH}")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {weight_path}!")
        return

    model.eval()  # Chế độ evaluation

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 3: CHUẨN BỊ DATA & DATALOADER
    # ─────────────────────────────────────────────────────────────
    img_dir = '../data/train2014'
    ann_file = '../data/annotations/COCO_amodal_train2014.json'
    
    # Batch size = 4 (4 ảnh cùng một lúc)
    BATCH_SIZE = 4 
    test_transform = A.Compose([A.Resize(224, 224)])
    dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 4: LẤY BATCH NGẪU NHIÊN & CHẠY
    # ─────────────────────────────────────────────────────────────
    inputs, target_masks, _, class_ids = next(iter(loader))
    inputs = inputs.to(DEVICE)
    class_ids = class_ids.to(DEVICE)
    print(f"🔄 Xử lý {inputs.shape[0]} ảnh cùng lúc...")

    with torch.no_grad():
        output_logits = model(inputs, class_ids)
        pred_masks = torch.sigmoid(output_logits)
        pred_masks = (pred_masks > 0.5).float()

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 5: TÁCH KÊNH & CHUYỂN SANG CPU
    # ─────────────────────────────────────────────────────────────
    imgs_rgb = inputs[:, :3, :, :].cpu() 
    visible_masks = inputs[:, 3, :, :].unsqueeze(1).cpu() 
    pred_masks = pred_masks.cpu() 
    target_masks = target_masks.unsqueeze(1).float().cpu() 

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 6: TẠO LƯỚI (GRID) ĐỂ HIỂN THỊ
    # ─────────────────────────────────────────────────────────────
    # make_grid: tạo lưới ảnh, nrow=2 → 2 ảnh mỗi hàng
    grid_rgb = make_grid(imgs_rgb, nrow=2, normalize=True, padding=8, pad_value=1)
    grid_visible = make_grid(visible_masks, nrow=2, normalize=False, padding=8, pad_value=1)
    grid_pred = make_grid(pred_masks, nrow=2, normalize=False, padding=8, pad_value=1)
    grid_target = make_grid(target_masks, nrow=2, normalize=False, padding=8, pad_value=1)

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 7: HIỂN THỊ KẾT QUẢ (4 CỘT)
    # ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # Cột 1: Ảnh RGB gốc
    axes[0].imshow(grid_rgb.permute(1, 2, 0))
    axes[0].set_title("Ảnh RGB Gốc", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Cột 2: Visible mask
    axes[1].imshow(grid_visible[0], cmap='gray')
    axes[1].set_title("Visible Mask", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Cột 3: Dự đoán Amodal
    axes[2].imshow(grid_pred[0], cmap='magma')
    axes[2].set_title("Dự Đoán Amodal (Model)", fontsize=14, fontweight='bold', color='darkblue')
    axes[2].axis('off')

    # Cột 4: Đáp án thực tế
    axes[3].imshow(grid_target[0], cmap='gray')
    axes[3].set_title("Đáp Án Thực Tế", fontsize=14, fontweight='bold')
    axes[3].axis('off')

    # Trang trí
    fig.patch.set_facecolor('#f5f5f5')
    plt.tight_layout(pad=2.0)
    plt.show()


if __name__ == "__main__":
    batch_test_pro()