def train():
    BATCH_SIZE = 4
    EPOCHS = 50
    RESUME_EPOCH = 0  # 🚨 CHÌA KHÓA NẰM Ở ĐÂY: Khai báo số Epoch đã học xong
    LEARNING_RATE = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {DEVICE}")

    img_dir = '../data/train2014'
    ann_file = '../data/annotations/COCO_amodal_train2014.json'

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    print("Đang chuẩn bị DataLoader với Data Augmentation...")
    train_dataset = AmodalDataset(img_dir=img_dir, ann_file=ann_file, transform=train_transform)
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

    os.makedirs('../checkpoints', exist_ok=True)
    
    print(f"\n🔥 TIẾP TỤC HUẤN LUYỆN TỪ EPOCH {RESUME_EPOCH + 1} ĐẾN {EPOCHS} 🔥")
    # Vòng lặp sẽ chạy từ số 10 -> 49 (Tức là Epoch 11 đến 50)
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