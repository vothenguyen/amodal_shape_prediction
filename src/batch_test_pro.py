import os
import torch
import random
import matplotlib.pyplot as plt
import albumentations as A
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

# Import từ file của tụi mình
from model import AmodalSwinUNet
from dataset import AmodalDataset

def batch_test_pro():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 RTX ĐÃ VÀO VỊ TRÍ, CHẠY TRÊN: {DEVICE}")

    # Gọi mô hình PRO MAX (Có Bơm Nhãn)
    model = AmodalSwinUNet(num_classes=91).to(DEVICE)
    
    TEST_EPOCH = 30
    weight_path = f'../checkpoints/swin_amodal_epoch_{TEST_EPOCH}.pth'
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"✅ Đã nạp thành công bộ não CHÚA TỂ Epoch {TEST_EPOCH}!")
    except FileNotFoundError:
        print(f"❌ Chưa tìm thấy file {weight_path}. Cậu nhớ tải từ Drive về thư mục checkpoints nha!")
        return

    model.eval()

    img_dir = '../data/train2014'
    ann_file = '../data/annotations/COCO_amodal_train2014.json'
    
    # 🚨 Đổi Batch Size thành 4 để lưới ảnh bự lên tối đa
    BATCH_SIZE = 4 
    test_transform = A.Compose([A.Resize(224, 224)]) 
    dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    inputs, target_masks, _, class_ids = next(iter(loader))
    inputs = inputs.to(DEVICE)
    class_ids = class_ids.to(DEVICE)
    print(f"Đang xử lý song song {inputs.shape[0]} vật thể...")

    with torch.no_grad():
        output_logits = model(inputs, class_ids)
        pred_masks = torch.sigmoid(output_logits)
        pred_masks = (pred_masks > 0.5).float() 

    # TÁCH KÊNH
    imgs_rgb = inputs[:, :3, :, :].cpu() 
    visible_masks = inputs[:, 3, :, :].unsqueeze(1).cpu() 
    # (Tạm ẩn Edge Mask khỏi khâu vẽ để tiết kiệm diện tích)
    pred_masks = pred_masks.cpu() 
    target_masks = target_masks.unsqueeze(1).float().cpu() 

    # 🚨 TẠO LƯỚI 2x2 BÊN TRONG TỪNG Ô (nrow=2)
    grid_rgb = make_grid(imgs_rgb, nrow=2, normalize=True, padding=8, pad_value=1)
    grid_visible = make_grid(visible_masks, nrow=2, normalize=False, padding=8, pad_value=1)
    grid_pred = make_grid(pred_masks, nrow=2, normalize=False, padding=8, pad_value=1)
    grid_target = make_grid(target_masks, nrow=2, normalize=False, padding=8, pad_value=1)

    # 🚨 ĐỔI BỐ CỤC THÀNH KHUNG CỬA SỔ 2x2 SIÊU TO (figsize=20,20)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten() # Duỗi mảng 2D thành 1D [0, 1, 2, 3] cho dễ gọi
    
    # Ô 1: Ảnh Gốc
    axes[0].imshow(grid_rgb.permute(1, 2, 0))
    axes[0].set_title("1. Ảnh Gốc (RGB)", fontsize=22, fontweight='bold')
    axes[0].axis('off')

    # Ô 2: Phần Nhìn Thấy
    axes[1].imshow(grid_visible[0], cmap='gray')
    axes[1].set_title("2. Phần Bị Khuất (Visible Mask)", fontsize=22, fontweight='bold')
    axes[1].axis('off')

    # Ô 3: AI Vẽ
    axes[2].imshow(grid_pred[0], cmap='magma') 
    axes[2].set_title("3. AI Tưởng Tượng (Pro Max)", fontsize=22, fontweight='bold', color='darkblue')
    axes[2].axis('off')

    # Ô 4: Đáp án
    axes[3].imshow(grid_target[0], cmap='gray')
    axes[3].set_title("4. Đáp Án Thực Tế (Target)", fontsize=22, fontweight='bold')
    axes[3].axis('off')

    fig.patch.set_facecolor('#f0f0f0') # Nền xám nhạt cho sang trọng
    plt.tight_layout(pad=3.0) # Tạo khoảng cách giữa các ô
    plt.show()

if __name__ == "__main__":
    batch_test_pro()