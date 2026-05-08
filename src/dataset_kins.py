import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class KINSDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transform = transform
        
        self.valid_anns = []
        # TÌM ĐÚNG TỪ KHÓA 'a_segm' VÀ 'i_segm' CỦA KINS
        for ann_id in self.coco.getAnnIds():
            ann = self.coco.loadAnns(ann_id)[0]
            if 'a_segm' in ann and 'i_segm' in ann:
                self.valid_anns.append(ann)
                    
        print(f"🔥 Bóc tách KINS thành công: {len(self.valid_anns)} vật thể!")

    def get_mask(self, ann, key):
        # 'Đánh lừa' pycocotools
        temp_ann = ann.copy()
        temp_ann['segmentation'] = temp_ann[key]
        return self.coco.annToMask(temp_ann)

    def __len__(self):
        return len(self.valid_anns)

    def __getitem__(self, idx):
        ann = self.valid_anns[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None: 
            img_path = img_path.replace('.jpg', '.png').replace('.png', '.jpg')
            image = cv2.imread(img_path)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        amodal_mask = self.get_mask(ann, 'a_segm')
        visible_mask = self.get_mask(ann, 'i_segm')
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(visible_mask, kernel, iterations=1)
        eroded = cv2.erode(visible_mask, kernel, iterations=1)
        edge_mask = (dilated - eroded)
        
        occluded_mask = np.clip(amodal_mask - visible_mask, 0, 1)
        
        if self.transform:
            augmented = self.transform(
                image=image, 
                masks=[amodal_mask, visible_mask, edge_mask, occluded_mask]
            )
            image = augmented['image']
            amodal_mask, visible_mask, edge_mask, occluded_mask = augmented['masks']
            
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)
        
        input_5ch = torch.cat([
            torch.from_numpy(image).float(), 
            torch.from_numpy(visible_mask).float().unsqueeze(0), 
            torch.from_numpy(edge_mask).float().unsqueeze(0)
        ], dim=0)
        
        class_id = torch.tensor(ann['category_id'], dtype=torch.long)
        
        return input_5ch, torch.from_numpy(amodal_mask).float(), torch.from_numpy(occluded_mask).float(), class_id
