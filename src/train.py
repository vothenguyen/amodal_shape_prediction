import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Import từ file của tụi mình
from dataset import AmodalDataset
from model import AmodalSwinUNet


class OcclusionAwareLoss(nn.Module):
    def __init__(self, occlusion_weight=5.0):  # THƯỞNG X5 ĐIỂM!!!
        super().__init__()
        # Để reduction='none' để lát mình tự nhân trọng số từng pixel
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        # 1. Chấm điểm BCE cho từng điểm ảnh
        bce_loss = self.bce(pred, target)

        # 2. LUẬT MỚI: Ma trận trọng số
        weight_matrix = torch.ones_like(target)  # Mặc định mọi pixel nhân 1
        # Những pixel nào nằm trong vùng bị khuất -> Đẩy trọng số lên x5!
        weight_matrix[occluded_region > 0.5] = self.occlusion_weight

        # 3. Nhân trọng số vào Loss và tính trung bình
        weighted_bce = (bce_loss * weight_matrix).mean()

        # 4. Dice Loss (Vẫn giữ để tổng thể mượt mà)
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()


def train():
    BATCH_SIZE = 4
    EPOCHS = 15
    RESUME_EPOCH = 0  # Bắt đầu lại từ con số 0 với não U-Net mới
    LEARNING_RATE = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {DEVICE}")

    img_dir = "../data/train2014"
    ann_file = "../data/annotations/COCO_amodal_train2014.json"

    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    print("Đang chuẩn bị DataLoader với Data Augmentation...")
    train_dataset = AmodalDataset(
        img_dir=img_dir, ann_file=ann_file, transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AmodalSwinUNet().to(DEVICE)

    # --- THUẬT HỒI SINH (RESUME) ---
    if RESUME_EPOCH > 0:
        weight_path = f"../checkpoints/swin_amodal_epoch_{RESUME_EPOCH}.pth"
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"\n🔄 HỒI SINH THÀNH CÔNG: Đã nạp lại 'bộ não' từ Epoch {RESUME_EPOCH}!")
    # -------------------------------

    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("../checkpoints", exist_ok=True)

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN TỪ EPOCH {RESUME_EPOCH + 1} ĐẾN {EPOCHS} 🔥")
    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for inputs, targets, occluded_region in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)
            occluded_region = occluded_region.unsqueeze(1).float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, occluded_region)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Kết thúc Epoch {epoch+1} | Trung bình Loss: {avg_loss:.4f}")

        save_path = f"../checkpoints/swin_amodal_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Đã lưu model tại: {save_path}\n")


if __name__ == "__main__":
    train()
