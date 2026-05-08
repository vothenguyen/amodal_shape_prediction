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
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        bce_loss = self.bce(pred, target)
        weight_matrix = torch.ones_like(target) 
        weight_matrix[occluded_region > 0.5] = self.occlusion_weight
        weighted_bce = (bce_loss * weight_matrix).mean()

        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()

def train():
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4  # Batch 4 x 4 = Batch ảo 16
    EPOCHS = 30
    RESUME_EPOCH = 0
    LEARNING_RATE = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {DEVICE}")

    # Đường dẫn chuẩn Kaggle Input
    img_dir = "/kaggle/input/datasets/anupammajhi/kitti-2d-object-detection/data_object_image_2/training/image_2"
    ann_file = "/kaggle/input/datasets/ryanthenguyen/kins-data/update_train_2020.json"

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        # Dùng Affine thay cho ShiftScaleRotate để tránh Warning của bản mới
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    print("Đang chuẩn bị DataLoader với Data Augmentation...")
    train_dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo mô hình với 8 lớp (KINS)
    model = AmodalSwinUNet(num_classes=8).to(DEVICE)

    if RESUME_EPOCH > 0:
        weight_path = f"/kaggle/working/checkpoints/swin_amodal_KINS_epoch_{RESUME_EPOCH}.pth"
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"\n🔄 HỒI SINH THÀNH CÔNG: Đã nạp lại 'bộ não' từ Epoch {RESUME_EPOCH}!")

    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Đảm bảo lưu vào đúng /kaggle/working/ để không bị lỗi phân quyền và dễ tải về
    SAVE_DIR = "/kaggle/working/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN KINS (AUTONOMOUS DRIVING) TỪ EPOCH {RESUME_EPOCH + 1} ĐẾN {EPOCHS} 🔥")
    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad() 

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (inputs, targets, occluded, class_ids) in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.unsqueeze(1).float().to(DEVICE)
            occluded = occluded.unsqueeze(1).float().to(DEVICE)
            class_ids = class_ids.to(DEVICE)

            outputs = model(inputs, class_ids) 
            loss = criterion(outputs, targets, occluded)
            
            loss = loss / ACCUMULATION_STEPS 
            loss.backward()

            if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        scheduler.step() 

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"✅ Kết thúc Epoch {epoch+1} | Trung bình Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # Tên file mới, chuẩn hóa cho KINS
        save_path = os.path.join(SAVE_DIR, f"swin_amodal_KINS_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Đã lưu model tại: {save_path}\n")

if __name__ == "__main__":
    train()
