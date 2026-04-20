import torch
import random
import matplotlib.pyplot as plt
import albumentations as A

# Import từ file của tụi mình
from model import AmodalSwinUNet
from dataset import AmodalDataset


def test_model():
    # 1. Cài đặt chạy trên CPU (Máy nhà chạy 1 ảnh dư sức!)
    DEVICE = torch.device("cpu")
    print("Đang khởi động AI trên CPU...")

    # 2. Gọi mô hình và lắp 'não' (Epoch 30) vào
    model = AmodalSwinUNet().to(DEVICE)
    weight_path = "../checkpoints/swin_amodal_epoch_30.pth"

    # Ép nó load bằng CPU kẻo nó tìm card đồ họa rồi báo lỗi
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()  # Bật chế độ đi thi (không học thêm nữa)

    # 3. Chuẩn bị dữ liệu (Lấy ảnh ở máy cậu)
    img_dir = "../data/train2014"
    ann_file = "../data/annotations/COCO_amodal_train2014.json"

    # Lưu ý: Lúc test thì KHÔNG lật hay xoay ảnh nữa, chỉ Resize cho chuẩn kích thước
    test_transform = A.Compose([A.Resize(224, 224)])
    dataset = AmodalDataset(
        img_dir=img_dir, ann_file=ann_file, transform=test_transform
    )

    # 4. Bốc đại 1 vật thể ngẫu nhiên để Test
    idx = random.randint(0, len(dataset) - 1)
    input_tensor, target_mask, occluded, class_id = dataset[idx]

    # Đóng gói thành Batch (1, 5, 224, 224) và class_id
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    class_id_batch = torch.tensor([class_id]).to(DEVICE)

    # 5. YÊU CẦU AI TƯỞNG TƯỞNG VÀ VẼ
    print(f"Đang xử lý ảnh thứ {idx}... Class ID: {class_id}")
    with torch.no_grad():
        output_logits = model(input_batch, class_id_batch)
        pred_mask = torch.sigmoid(output_logits)
        pred_mask = (pred_mask > 0.5).squeeze().numpy()  # Ép về nhị phân (Đen/Trắng)

    # 6. Tách Kênh & Triển Lãm
    img_rgb = input_tensor[:3].numpy().transpose(1, 2, 0)
    visible_mask = input_tensor[3].numpy()
    truth_mask = target_mask.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Ảnh Gốc (RGB)")
    axes[0].axis("off")

    axes[1].imshow(visible_mask, cmap="gray")
    axes[1].set_title("2. Kênh 4 (Phần nhìn thấy)")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("3. U-Net Tưởng Tượng (Epoch 30)")
    axes[2].axis("off")

    axes[3].imshow(truth_mask, cmap="gray")
    axes[3].set_title("4. Đáp án Thực Tế")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_model()
