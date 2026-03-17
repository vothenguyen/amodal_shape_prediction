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


# ==========================================
# VŨ KHÍ 1: HÀM CHẤM ĐIỂM KÉP (BCE + DICE LOSS)
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + 1e-6) / (
            pred_flat.sum() + target_flat.sum() + 1e-6
        )
        dice_loss = 1.0 - dice_score
        return bce_loss + dice_loss


def train():
    BATCH_SIZE = 4
    EPOCHS = 50
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

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("../checkpoints", exist_ok=True)

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN TỪ EPOCH {RESUME_EPOCH + 1} ĐẾN {EPOCHS} 🔥")
    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for inputs, targets in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
