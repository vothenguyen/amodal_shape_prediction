"""
===================================================================================
DATASET KINS - Xử lý dữ liệu KITTI In-house (KINS Amodal)
===================================================================================
Dataset class để tải và xử lý dữ liệu amodal từ KINS annotation files.

Tính năng:
- Đọc đa giác amodal mask từ khóa 'a_segm'
- Đọc đa giác visible mask từ khóa 'i_segm'
- Sinh ranh giới edge mask bằng toán tử hình thái học (cv2.dilate - cv2.erode)
- Tính toán vùng bị che khuất occluded_mask = clip(amodal - visible, 0, 1)
- Ghép tensor 5 kênh: RGB (3) + Visible (1) + Edge (1)
===================================================================================
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class KINSDataset(Dataset):
    """
    Dataset class cho Amodal Shape Prediction trên KINS (KITTI In-house).
    
    Args:
        img_dir: Đường dẫn thư mục ảnh KITTI (ví dụ: data_object_image_2/training/image_2)
        ann_file: Đường dẫn file JSON annotation KINS (ví dụ: update_train_2020.json)
        transform: Pipeline biến đổi Albumentations
    """
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transform = transform

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        print(f"📂 Đang nạp file annotation KINS: {ann_file}...")
        self.coco = COCO(ann_file)

        self.valid_anns = []
        for ann_id in self.coco.getAnnIds():
            ann = self.coco.loadAnns(ann_id)[0]
            if 'a_segm' in ann and 'i_segm' in ann:
                self.valid_anns.append(ann)

        print(f"✅ Bóc tách KINS thành công: {len(self.valid_anns):,} vật thể hợp lệ!")

    def get_mask(self, ann, key):
        temp_ann = ann.copy()
        temp_ann['segmentation'] = ann[key]
        return self.coco.annToMask(temp_ann)

    def __len__(self):
        return len(self.valid_anns)

    def __getitem__(self, idx):
        ann = self.valid_anns[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            # Fallback nếu định dạng ảnh khác .png / .jpg
            alt_path = img_path.replace('.jpg', '.png') if img_path.endswith('.jpg') else img_path.replace('.png', '.jpg')
            image = cv2.imread(alt_path)

        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        amodal_mask = self.get_mask(ann, 'a_segm')
        visible_mask = self.get_mask(ann, 'i_segm')

        # Toán tử hình thái sinh edge mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_mask = cv2.dilate(visible_mask, kernel, iterations=1) - cv2.erode(visible_mask, kernel, iterations=1)

        # Tính occluded mask
        occluded_mask = np.clip(amodal_mask.astype(np.int32) - visible_mask.astype(np.int32), 0, 1).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(
                image=image,
                masks=[visible_mask, edge_mask, amodal_mask, occluded_mask]
            )
            image = augmented['image']
            visible_mask = augmented['masks'][0]
            edge_mask = augmented['masks'][1]
            amodal_mask = augmented['masks'][2]
            occluded_mask = augmented['masks'][3]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        input_5ch = torch.cat([
            image,
            torch.from_numpy(visible_mask).float().unsqueeze(0),
            torch.from_numpy(edge_mask).float().unsqueeze(0)
        ], dim=0)

        class_id = torch.tensor(ann.get('category_id', ann.get('category', 0)), dtype=torch.long)

        return (
            input_5ch,
            torch.from_numpy(amodal_mask).float(),
            torch.from_numpy(occluded_mask).float(),
            class_id
        )
