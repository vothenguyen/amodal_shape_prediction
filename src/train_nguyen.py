import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from src.model_Nguyen import AmodalPipelineNguyen
from src.dataset_nguyen import AmodalDatasetNguyen
from src.loss_nguyen import MultiTaskAmodalLoss

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        images = batch['image'].to(device)
        GT_v = batch['visible_mask'].to(device)
        GT_b = batch['boundary_mask'].to(device)
        GT_a = batch['amodal_mask'].to(device)
        cat_ids = batch['category_id'].to(device)
        
        optimizer.zero_grad()
        
        preds = model(images, cat_ids)
        targets = (GT_v, GT_b, GT_a)
        
        loss_dict = criterion(preds, targets)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Dataloader
    dataset = AmodalDatasetNguyen(root_dir=args.data_dir, split='train')
    
    # Check if dataset is empty to prevent DataLoader crash
    if len(dataset) == 0:
        raise RuntimeError(f"Error: Dataset is empty. No images found in {os.path.join(args.data_dir, 'train2014')}")
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:
        print("Warning: Dataloader is empty, check dataset path.")
        
    # Model
    model = AmodalPipelineNguyen(num_classes=91).to(device)
    
    # Loss & Optimizer
    criterion = MultiTaskAmodalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(args.epochs):
        if len(loader) == 0: break
        
        epoch_loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoints/nguyen_model_epoch_{epoch+1}.pth')

    print("Training complete!")

if __name__ == '__main__':
    main()
