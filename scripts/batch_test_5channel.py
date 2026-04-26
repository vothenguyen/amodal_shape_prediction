"""
===================================================================================
BATCH TEST 5-CHANNEL - Visualize tất cả 5 kênh của mô hình
===================================================================================
Script để kiểm tra đầu vào/đầu ra của mô hình với đầy đủ 5 kênh.

Hiển thị:
1. Ảnh RGB gốc (3 kênh)
2. Visible mask (kênh 4)
3. Edge mask (kênh 5)
4. Dự đoán Amodal từ mô hình
5. Đáp án thực tế

Batch size: 16 ảnh, hiển thị dạng lưới 4×4

Chạy: python src/batch_test_5channel.py
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


def batch_test_5channel():
    """
    Chạy test trên batch 16 ảnh với hiển thị đầy đủ 5 kênh.
    """
    # ─────────────────────────────────────────────────────────────
    # KHỞI ĐỘNG
    # ─────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chạy trên thiết bị: {DEVICE}")

    # ─────────────────────────────────────────────────────────────
    # NẠP MÔ HÌNH
    # ─────────────────────────────────────────────────────────────
    model = AmodalSwinUNet().to(DEVICE)
    
    TEST_EPOCH = 15 
    weight_path = f'../checkpoints/swin_amodal_epoch_{TEST_EPOCH}.pth'
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"✅ Đã nạp mô hình Epoch {TEST_EPOCH}")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy {weight_path}")
        return

    model.eval()

    # ─────────────────────────────────────────────────────────────
    # CHUẨN BỊ DATA
    # ─────────────────────────────────────────────────────────────
    img_dir = '../data/train2014'
    ann_file = '../data/annotations/COCO_amodal_train2014.json'
    
    BATCH_SIZE = 16  # 16 ảnh → lưới 4×4
    test_transform = A.Compose([A.Resize(224, 224)])
    dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ─────────────────────────────────────────────────────────────
    # LẤY BATCH & CHẠY
    # ─────────────────────────────────────────────────────────────
    inputs, target_masks, _ = next(iter(loader))
    inputs = inputs.to(DEVICE)
    print(f"🔄 Xử lý batch {inputs.shape[0]} ảnh (lưới 4×4)...")

    with torch.no_grad():
        output_logits = model(inputs)
        pred_masks = torch.sigmoid(output_logits)
        pred_masks = (pred_masks > 0.5).float()

    # ─────────────────────────────────────────────────────────────
    # TÁCH 5 KÊNH
    # ─────────────────────────────────────────────────────────────
    imgs_rgb = inputs[:, :3, :, :].cpu()          # Kênh 0-2: RGB
    visible_masks = inputs[:, 3, :, :].unsqueeze(1).cpu()  # Kênh 3: Visible
    edge_masks = inputs[:, 4, :, :].unsqueeze(1).cpu()     # Kênh 4: Edge
    pred_masks = pred_masks.cpu() 
    target_masks = target_masks.unsqueeze(1).float().cpu()

    # ─────────────────────────────────────────────────────────────
    # TẠO LƯỚI 4×4
    # ─────────────────────────────────────────────────────────────
    grid_rgb = make_grid(imgs_rgb, nrow=4, normalize=True, padding=5, pad_value=1)
    grid_visible = make_grid(visible_masks, nrow=4, normalize=False, padding=5, pad_value=1)
    grid_edge = make_grid(edge_masks, nrow=4, normalize=False, padding=5, pad_value=1)
    grid_pred = make_grid(pred_masks, nrow=4, normalize=False, padding=5, pad_value=1)
    grid_target = make_grid(target_masks, nrow=4, normalize=False, padding=5, pad_value=1)

    # ─────────────────────────────────────────────────────────────
    # HIỂN THỊ (5 CỘT)
    # ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(30, 8))
    
    # Cột 1: RGB
    axes[0].imshow(grid_rgb.permute(1, 2, 0))
    axes[0].set_title("RGB Gốc", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Cột 2: Visible mask
    axes[1].imshow(grid_visible[0], cmap='gray')
    axes[1].set_title("Kênh 3 (Visible Mask)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Cột 3: Edge mask
    axes[2].imshow(grid_edge[0], cmap='gray')
    axes[2].set_title("Kênh 4 (Edge Mask)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Cột 4: Dự đoán
    axes[3].imshow(grid_pred[0], cmap='gray')
    axes[3].set_title("Dự Đoán Amodal", fontsize=14, fontweight='bold', color='blue')
    axes[3].axis('off')

    # Cột 5: Đáp án
    axes[4].imshow(grid_target[0], cmap='gray')
    axes[4].set_title("Đáp Án Thực Tế", fontsize=14, fontweight='bold')
    axes[4].axis('off')

    # Trang trí
    fig.patch.set_facecolor('#e8e8e8')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    batch_test_5channel()