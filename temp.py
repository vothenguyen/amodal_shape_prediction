import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import sys

# Thêm thư mục hiện tại vào sys.path để import từ scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scripts.dataset import AmodalDataset
from scripts.model import AmodalSwinUNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Đường dẫn file
    img_dir = "data/val2014/val2014"
    ann_file = "data/annotations/annotations/COCO_amodal_val2014.json"
    checkpoint_path = "checkpoints/swin_amodal_epoch_30.pth"

    # 1. Khởi tạo dataset
    print("Khởi tạo Dataset...")
    transform = A.Compose([A.Resize(224, 224)])
    try:
        dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=transform)
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        return

    # 2. Khởi tạo mô hình và tải trọng số
    print("Khởi tạo mô hình AmodalSwinUNet...")
    model = AmodalSwinUNet().to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Tải trọng số từ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle possible nested states (e.g. if saved with 'model_state_dict')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Không tìm thấy checkpoint tại {checkpoint_path}, sử dụng mô hình chưa huấn luyện.")
    
    model.eval()

    # 3. Tìm một mẫu có vùng bị che khuất (occlusion) để hiển thị cho trực quan
    print("Tìm mẫu ngẫu nhiên có vật thể bị che khuất...")
    import random
    sample_idx = 0
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for i in indices:
        _, _, occluded_region, _ = dataset[i]
        if occluded_region.sum() > 50:  # Chọn mẫu có vùng bị che > 50 pixel
            sample_idx = i
            break
            
    print(f"Đang dự đoán trên mẫu thứ {sample_idx}...")
    input_tensor, amodal_tensor, occluded_region, cat_id = dataset[sample_idx]

    # 4. Dự đoán hình dáng toàn bộ (Amodal shape)
    input_batch = input_tensor.unsqueeze(0).to(device)
    cat_id_batch = cat_id.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logits = model(input_batch, cat_id_batch)
        # Áp dụng sigmoid và threshold 0.5 để lấy binary mask
        pred_mask = (torch.sigmoid(pred_logits[0, 0]) > 0.5).cpu().numpy()

    # 5. Tách các thành phần từ input tensor để hiển thị
    # Kênh 0-2: RGB (đang ở dạng [3, H, W], chuyển về [H, W, 3])
    rgb_image = input_tensor[0:3].permute(1, 2, 0).numpy()
    
    # Kênh 3: Visible mask (Dạng nhị phân 0-1)
    visible_mask = input_tensor[3].numpy()
    
    # Kênh 4: Edge mask (Dạng nhị phân 0-1)
    edge_mask = input_tensor[4].numpy()

    # 6. Tạo ảnh kết quả: RGB + lớp mờ (overlay) của amodal mask
    result_overlay = rgb_image.copy()
    
    # Lớp phủ màu đỏ cho amodal mask (với độ mờ 0.5)
    color = np.array([1.0, 0.0, 0.0]) # Màu đỏ
    alpha = 0.5 # Độ mờ
    
    # Áp dụng overlay tại những nơi có amodal mask
    for c in range(3):
        result_overlay[:, :, c] = np.where(
            pred_mask,
            result_overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            result_overlay[:, :, c]
        )

    # Đảm bảo giá trị pixel nằm trong khoảng [0, 1]
    result_overlay = np.clip(result_overlay, 0, 1)

    # 7. Hiển thị 4 hình ảnh
    plt.figure(figsize=(16, 4))
    plt.suptitle("Amodal Shape Prediction", fontsize=16)

    # Ảnh 1: RGB ban đầu
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_image)
    plt.title("RGB ban đầu")
    plt.axis("off")

    # Ảnh 2: Visible Mask
    plt.subplot(1, 4, 2)
    plt.imshow(visible_mask, cmap='gray')
    plt.title("Visible Mask")
    plt.axis("off")

    # Ảnh 3: Edge Mask
    plt.subplot(1, 4, 3)
    plt.imshow(edge_mask, cmap='gray')
    plt.title("Edge Mask")
    plt.axis("off")

    # Ảnh 4: Result Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(result_overlay)
    plt.title("RGB + Amodal Mask (Pred)")
    plt.axis("off")

    plt.tight_layout()
    
    # Lưu hình ảnh và hiển thị
    save_path = "results/demo_output.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu kết quả tại: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()