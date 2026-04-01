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

def batch_test_5channel():
    # 1. Khởi động quái vật RTX 3050
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 RTX 3050 ĐÃ VÀO VỊ TRÍ, CHẠY TRÊN: {DEVICE}")

    # 2. Gọi mô hình 5 KÊNH
    model = AmodalSwinUNet().to(DEVICE)
    
    # LƯU Ý: Cậu canh xem Colab nó chạy tới Epoch mấy thì sửa số ở đây nha (ví dụ: epoch_10)
    TEST_EPOCH = 15 
    weight_path = f'../checkpoints/swin_amodal_epoch_{TEST_EPOCH}.pth'
    # cdC:\Users\THE NGUYEN\OneDrive - 7jxd6p\Desktop\uit\amodal_shape_project\checkpoints\swin_amodal_epoch_15.pth
    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"✅ Đã nạp thành công bộ não Pro Max Epoch {TEST_EPOCH}!")
    except FileNotFoundError:
        print(f"❌ Ôi thôi, chưa tìm thấy file {weight_path}. Cậu nhớ tải từ Drive về máy tính bỏ vào thư mục checkpoints nha!")
        return

    model.eval() # Bật chế độ đi thi

    # 3. Chuẩn bị dữ liệu
    img_dir = '../data/train2014'
    ann_file = '../data/annotations/COCO_amodal_train2014.json'
    
    BATCH_SIZE = 16 
    test_transform = A.Compose([A.Resize(224, 224)]) # Đi thi chỉ Resize, không lật xoay
    dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=test_transform)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. BỐC MỘT LƯỢT 16 ẢNH (Bây giờ dataset trả về 3 món)
    inputs, target_masks, _ = next(iter(loader))
    inputs = inputs.to(DEVICE)
    print(f"Đang xử lý song song {inputs.shape[0]} vật thể 5 KÊNH trên GPU...")

    # 5. YÊU CẦU AI TƯỞNG TƯỢNG VÀ VẼ
    with torch.no_grad():
        output_logits = model(inputs)
        pred_masks = torch.sigmoid(output_logits)
        pred_masks = (pred_masks > 0.5).float() # Ép về nhị phân

    # 6. TÁCH 5 KÊNH ĐỂ ĐEM ĐI TRIỂN LÃM
    imgs_rgb = inputs[:, :3, :, :].cpu() 
    visible_masks = inputs[:, 3, :, :].unsqueeze(1).cpu() 
    edge_masks = inputs[:, 4, :, :].unsqueeze(1).cpu() # Kênh 5 mới ra lò!
    pred_masks = pred_masks.cpu() 
    target_masks = target_masks.unsqueeze(1).float().cpu() 

    # Dùng make_grid để vẽ lưới, chia viền trắng (padding=5, pad_value=1)
    grid_rgb = make_grid(imgs_rgb, nrow=4, normalize=True, padding=5, pad_value=1)
    grid_visible = make_grid(visible_masks, nrow=4, normalize=False, padding=5, pad_value=1)
    grid_edge = make_grid(edge_masks, nrow=4, normalize=False, padding=5, pad_value=1)
    grid_pred = make_grid(pred_masks, nrow=4, normalize=False, padding=5, pad_value=1)
    grid_target = make_grid(target_masks, nrow=4, normalize=False, padding=5, pad_value=1)

    # VẼ TẤT CẢ LÊN MỘT MÀN HÌNH (5 Cột)
    fig, axes = plt.subplots(1, 5, figsize=(30, 10))
    
    axes[0].imshow(grid_rgb.permute(1, 2, 0))
    axes[0].set_title("1. Ảnh Gốc (RGB)", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(grid_visible[0], cmap='gray')
    axes[1].set_title("2. Kênh 4 (Visible)", fontsize=16, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(grid_edge[0], cmap='gray')
    axes[2].set_title("3. Kênh 5 (Viền Gợi Ý)", fontsize=16, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(grid_pred[0], cmap='gray')
    axes[3].set_title("4. AI Tưởng Tượng (Pro Max)", fontsize=16, fontweight='bold', color='blue')
    axes[3].axis('off')

    axes[4].imshow(grid_target[0], cmap='gray')
    axes[4].set_title("5. Đáp án Thực Tế", fontsize=16, fontweight='bold')
    axes[4].axis('off')

    fig.patch.set_facecolor('#e0e0e0')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    batch_test_5channel()