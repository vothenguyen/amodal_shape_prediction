import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class AmodalDatasetNguyen(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, f'{split}2014')
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')] if os.path.exists(self.image_dir) else []
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image RGB
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
        # Mocking ground truth masks for training purpose
        # In reality, this requires parsing COCO Amodal annotations
        h, w = image.shape[:2]
        visible_mask = np.zeros((h, w), dtype=np.float32)
        boundary_mask = np.zeros((h, w), dtype=np.float32)
        amodal_mask = np.zeros((h, w), dtype=np.float32)
        
        # Add some random blocks just to make it run
        if h > 50 and w > 50:
            amodal_mask[h//4:3*h//4, w//4:3*w//4] = 1.0
            visible_mask[h//4:h//2, w//4:3*w//4] = 1.0
            boundary_mask = amodal_mask - visible_mask
            boundary_mask = np.clip(boundary_mask, 0, 1)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        visible_mask = torch.from_numpy(visible_mask).unsqueeze(0)
        boundary_mask = torch.from_numpy(boundary_mask).unsqueeze(0)
        amodal_mask = torch.from_numpy(amodal_mask).unsqueeze(0)
        
        # Random class id
        category_id = torch.tensor(1, dtype=torch.long)
        
        return {
            'image': image,
            'visible_mask': visible_mask,
            'boundary_mask': boundary_mask,
            'amodal_mask': amodal_mask,
            'category_id': category_id
        }
