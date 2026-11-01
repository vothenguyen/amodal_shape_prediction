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
- Tự động dò tìm đường dẫn ảnh KITTI trên các cấu trúc thư mục khác nhau
- Fallback an toàn chống lỗi thiếu ảnh hoặc sai lệch kích thước
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

    def _find_image_path(self, file_name):
        """Dò tìm đường dẫn ảnh KITTI trên các vị trí cấu trúc thư mục thường gặp."""
        candidates = [
            os.path.join(self.img_dir, file_name),
            os.path.join(self.img_dir, "testing", "image_2", file_name),
            os.path.join(self.img_dir, "training", "image_2", file_name),
            os.path.join(self.img_dir, "data_object_image_2", "testing", "image_2", file_name),
            os.path.join(self.img_dir, "data_object_image_2", "training", "image_2", file_name),
            f"/content/kitti_data/testing/image_2/{file_name}",
            f"/content/kitti_data/training/image_2/{file_name}",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
            # Hỗ trợ hoán đổi đuôi .jpg <-> .png
            alt = p.replace('.jpg', '.png') if p.endswith('.jpg') else p.replace('.png', '.jpg')
            if os.path.exists(alt):
                return alt
        return None

    def __getitem__(self, idx):
        ann = self.valid_anns[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        file_name = img_info['file_name']

        img_path = self._find_image_path(file_name)
        image = None
        if img_path is not None:
            image = cv2.imread(img_path)

        if image is None:
            # Fallback zero image nếu thiếu file trên đĩa để không bao giờ bị dừng tiến trình
            target_h = img_info.get('height', 375)
            target_w = img_info.get('width', 1242)
            image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        amodal_mask = self.get_mask(ann, 'a_segm')
        visible_mask = self.get_mask(ann, 'i_segm')

        # Đảm bảo kích thước ảnh và mask đồng bộ trước khi xử lý
        h_img, w_img = image.shape[:2]
        if visible_mask.shape != (h_img, w_img):
            visible_mask = cv2.resize(visible_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        if amodal_mask.shape != (h_img, w_img):
            amodal_mask = cv2.resize(amodal_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

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
