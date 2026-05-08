import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# SỬ DỤNG BỘ BÓC TÁCH MỚI CỦA KINS
from dataset_kins import KINSDataset
from model import AmodalSwinUNet

class OcclusionAwareLoss(nn.Module):
    def __init__(self, occlusion_weight=5.0):
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
    # 🚀 TỐI ƯU HÓA CHO A100 🚀
    BATCH_SIZE = 64        # Ép A100 ăn 64 ảnh cùng lúc
    ACCUMULATION_STEPS = 1 
    EPOCHS = 30
    RESUME_EPOCH = 0
    LEARNING_RATE = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên siêu xe: {torch.cuda.get_device_name(0)}")

    img_dir = "/content/kitti_data/training/image_2"
    ann_file = "/content/kins_data/update_train_2020.json"

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    print("Đang chuẩn bị DataLoader với Data Augmentation...")
    train_dataset = KINSDataset(img_dir=img_dir, ann_file=ann_file, transform=train_transform)
    
    # 🚀 TĂNG TỐC ĐỘ ĐỌC Ổ CỨNG 🚀
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,       
        pin_memory=True      
    )

    model = AmodalSwinUNet(num_classes=20, use_spatial_attention=True).to(DEVICE)
    SAVE_DIR = "../checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 🚀 BẢO BỐI KÍCH HOẠT TENSOR CORES (AMP) 🚀
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN KINS BẰNG A100 TỪ EPOCH {RESUME_EPOCH + 1} ĐẾN {EPOCHS} 🔥")
    for epoch in range(RESUME_EPOCH, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad() 

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (inputs, targets, occluded, class_ids) in progress_bar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(DEVICE, non_blocking=True)
            occluded = occluded.unsqueeze(1).float().to(DEVICE, non_blocking=True)
            class_ids = class_ids.to(DEVICE, non_blocking=True)

            # Ép GPU dùng FP16 để tính toán tốc độ ánh sáng
            with torch.cuda.amp.autocast():
                outputs = model(inputs, class_ids) 
                loss = criterion(outputs, targets, occluded)
            
            # Tính gradient bằng Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step() 
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Kết thúc Epoch {epoch+1} | Trung bình Loss: {avg_loss:.4f}")

        save_path = os.path.join(SAVE_DIR, f"swin_amodal_KINS_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Đã lưu model tại: {save_path}\n")

if __name__ == "__main__":
    train()
