"""
===================================================================================
AMODAL DATASET D2SA - Xử lý dữ liệu D2S-Amodal
===================================================================================
Dataset class để tải và xử lý dữ liệu amodal từ D2SA (MVTec D2S Amodal) annotation.

Khác biệt so với AmodalDataset (COCOA):
- D2SA dùng flat COCO-style: mỗi annotation = 1 instance (không có "regions" nested)
- Mask dùng RLE encoding (không phải polygon)
- visible_mask có sẵn trực tiếp (không cần tính từ "order")
- Category IDs: 1–60 (liên tục) → trừ 1 trước khi đưa vào nn.Embedding

Yêu cầu: pycocotools >= 2.0.11 (upgrade nếu cần: pip install --upgrade pycocotools)
===================================================================================
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils


def decode_rle(rle_dict):
    """
    Decode D2SA RLE mask.
    
    D2SA lưu RLE counts dạng string (COCO compressed RLE).
    pycocotools 2.0.11+ cần counts dạng bytes.
    
    Returns:
        np.ndarray: Binary mask (H, W), dtype=uint8, values {0, 1}
    """
    rle = rle_dict.copy()
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    return mask_utils.decode(rle)


class AmodalDatasetD2SA(Dataset):
    """
    Dataset class cho Amodal Shape Prediction trên D2S-Amodal.
    
    Cấu trúc dữ liệu D2SA:
    - Mỗi annotation = 1 instance (flat, không nested regions)
    - annotation["segmentation"]: RLE → amodal mask
    - annotation["visible_mask"]: RLE → visible mask (có sẵn)
    - annotation["invisible_mask"]: RLE → invisible mask (optional, chỉ khi bị che)
    - annotation["category_id"]: 1–60
    - annotation["occlude_rate"]: float, tỷ lệ bị che
    
    Output: Giống hệt AmodalDataset (COCOA) để dùng chung model/train loop.
    
    Args:
        img_dir: Đường dẫn thư mục chứa ảnh (data/D2SA/images/)
        ann_file: Đường dẫn file annotation JSON
        transform: Hàm augmentation từ Albumentations (tùy chọn)
    """
    
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        print(f"📂 Đang nạp file annotation D2SA {ann_file} vào bộ nhớ...")
        
        # Load annotation JSON trực tiếp (không qua COCO API vì D2SA có thêm
        # các field custom mà COCO API không index: visible_mask, invisible_mask)
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build image lookup: image_id → image info
        self.images = {img['id']: img for img in data['images']}
        
        # Build category lookup: category_id → category info
        self.categories = {cat['id']: cat for cat in data.get('categories', [])}
        
        # ──────────────────────────────────────────────────────────────────
        # BƯỚC QUAN TRỌNG: Mỗi annotation = 1 mẫu huấn luyện
        # ──────────────────────────────────────────────────────────────────
        # Không cần bóc tách regions vì D2SA đã flat sẵn.
        # Chỉ filter bỏ iscrowd=True (nếu có) và annotation không có mask.
        self.annotations = []
        skipped_crowd = 0
        skipped_no_mask = 0
        
        for ann in data['annotations']:
            # Skip crowd annotations (convention COCO)
            if ann.get('iscrowd', False):
                skipped_crowd += 1
                continue
            # Skip nếu thiếu segmentation hoặc visible_mask
            if 'segmentation' not in ann or 'visible_mask' not in ann:
                skipped_no_mask += 1
                continue
            self.annotations.append(ann)
        
        print(
            f"✅ Hoàn tất! {len(self.annotations)} mẫu"
            + (f" (bỏ {skipped_crowd} crowd)" if skipped_crowd else "")
            + (f" (bỏ {skipped_no_mask} thiếu mask)" if skipped_no_mask else "")
        )

    def __len__(self):
        """Trả về tổng số mẫu trong dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Lấy một mẫu từ dataset.
        
        Quy trình (đối chiếu 1:1 với AmodalDataset COCOA):
        1. Lấy thông tin annotation và ảnh
        2. Đọc ảnh RGB từ file
        3. Decode amodal mask từ RLE
        4. Decode visible mask từ RLE (có sẵn, không cần tính từ order)
        5. Data augmentation (nếu có)
        6. Tính edge mask (viền gợi ý)
        7. Kết hợp thành 5 kênh input tensor
        
        Returns:
            Tuple gồm 4 thành phần (giống COCOA):
            - input_tensor: Ảnh 5 kênh [5, H, W]
            - amodal_tensor: Amodal mask [H, W]
            - occluded_region: Vùng bị che [H, W]
            - cat_id: Class ID cho embedding (0-indexed: category_id - 1)
        """
        
        ann = self.annotations[idx]
        
        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 1: LẤY THÔNG TIN ẢNH
        # ──────────────────────────────────────────────────────────────────
        img_id = ann['image_id']
        img_info = self.images[img_id]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 2: ĐỌC ẢNH RGB
        # ──────────────────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB
        img_h, img_w = image.shape[:2]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 3: DECODE AMODAL MASK (từ RLE — toàn bộ vật thể)
        # ──────────────────────────────────────────────────────────────────
        # Size đọc từ RLE per-sample (KHÔNG hard-code — D2SA có 1556 kích thước khác nhau)
        amodal_mask = decode_rle(ann['segmentation'])  # (H_rle, W_rle), uint8
        
        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 4: DECODE VISIBLE MASK (từ RLE — có sẵn, không cần tính)
        # ──────────────────────────────────────────────────────────────────
        visible_mask = decode_rle(ann['visible_mask'])  # (H_rle, W_rle), uint8
        
        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 4b: XỬ LÝ KÍCH THƯỚC ẢNH vs MASK
        # ──────────────────────────────────────────────────────────────────
        # D2SA zero-pads ảnh khi object tràn ra ngoài ảnh gốc → mask có thể
        # lớn hơn ảnh gốc. Cần đồng bộ kích thước.
        mask_h, mask_w = amodal_mask.shape
        
        if (mask_h, mask_w) != (img_h, img_w):
            # Pad ảnh RGB cho khớp với mask (zero-padding, giống D2SA)
            padded_image = np.zeros((mask_h, mask_w, 3), dtype=image.dtype)
            # Đặt ảnh gốc ở góc trên trái (giống convention D2SA)
            padded_image[:img_h, :img_w, :] = image
            image = padded_image

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 5: DATA AUGMENTATION
        # ──────────────────────────────────────────────────────────────────
        if self.transform:
            # Áp dụng các phép biến đổi trên ảnh và cả 2 mask
            transformed = self.transform(image=image, masks=[amodal_mask, visible_mask])
            image = transformed['image']
            amodal_mask = transformed['masks'][0]
            visible_mask = transformed['masks'][1]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 6: CHUYỂN ĐỔI TỪ NUMPY → PYTORCH TENSOR
        # ──────────────────────────────────────────────────────────────────
        # Chuyển ảnh RGB: [H, W, 3] → [3, H, W] và chuẩn hóa về [0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Chuyển amodal mask: [H, W] → [H, W] (vẫn là 2D)
        amodal_tensor = torch.from_numpy(amodal_mask).float()
        # Chuyển visible mask: [H, W] → [H, W]
        visible_tensor = torch.from_numpy(visible_mask).float()

        # Tính vùng bị che khuất: amodal - visible (bằng 0 nếu không bị che)
        occluded_region = torch.clamp(amodal_tensor - visible_tensor, min=0.0)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 7: VẼ EDGE MASK (Viền gợi ý)
        # ──────────────────────────────────────────────────────────────────
        # Edge mask = biên của visible mask (dilation - erosion)
        # Giống hệt COCOA dataset
        visible_uint8 = (visible_mask * 255).astype(np.uint8)
        # Kernel 5×5 để tìm cạnh
        kernel = np.ones((5, 5), np.uint8)
        # Dilation: thêm white pixels xung quanh ranh giới
        dilation = cv2.dilate(visible_uint8, kernel, iterations=1)
        # Erosion: bớt white pixels từ ranh giới
        erosion = cv2.erode(visible_uint8, kernel, iterations=1)
        # Edge = sự chênh lệch (ranh giới giữa dilate và erode)
        edge_mask = torch.tensor((dilation - erosion) / 255.0, dtype=torch.float32)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 8: KẾT HỢP THÀNH 5 KÊNH INPUT
        # ──────────────────────────────────────────────────────────────────
        # Kênh 0-2: RGB ảnh gốc
        # Kênh 3: Visible mask (chỉ phần nhìn thấy)
        # Kênh 4: Edge mask (viền gợi ý)
        input_tensor = torch.cat([
            image_tensor,                          # Kênh 0-2: RGB [3, H, W]
            visible_tensor.unsqueeze(0),           # Kênh 3: Visible [1, H, W]
            edge_mask.unsqueeze(0)                 # Kênh 4: Edge [1, H, W]
        ], dim=0)  # Nối theo chiều kênh → [5, H, W]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 9: CATEGORY ID — Off-by-one correction
        # ──────────────────────────────────────────────────────────────────
        # D2SA category_id: 1–60
        # nn.Embedding(60, 768): index 0–59
        # → Trừ 1 để chuyển sang 0-indexed
        raw_cat_id = ann['category_id']
        cat_id = raw_cat_id - 1  # 0-indexed cho nn.Embedding
        
        # Assert để bắt lỗi sớm nếu category_id ngoài phạm vi
        assert 0 <= cat_id < 60, (
            f"category_id {raw_cat_id} ngoài phạm vi [1, 60] — "
            f"ann_id={ann['id']}, img_id={img_id}"
        )

        return input_tensor, amodal_tensor, occluded_region, torch.tensor(cat_id, dtype=torch.long)


class AmodalDatasetD2SA_Concat(AmodalDatasetD2SA):
    """
    Dataset gộp nhiều file annotation JSON (ví dụ: training_rot0 + augmented).
    
    Theo paper D2SA gốc, training set = training_rot0 + augmented.
    Class này tải và gộp annotations từ nhiều file.
    
    Args:
        img_dir: Đường dẫn thư mục chứa ảnh
        ann_files: List các đường dẫn file annotation JSON
        transform: Hàm augmentation từ Albumentations (tùy chọn)
    """
    
    def __init__(self, img_dir, ann_files, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = {}
        self.categories = {}
        self.annotations = []
        
        total_skipped_crowd = 0
        total_skipped_no_mask = 0
        
        for ann_file in ann_files:
            print(f"📂 Đang nạp {ann_file}...")
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Merge images
            for img in data['images']:
                self.images[img['id']] = img
            
            # Merge categories (should be same across files)
            for cat in data.get('categories', []):
                self.categories[cat['id']] = cat
            
            # Filter and add annotations
            for ann in data['annotations']:
                if ann.get('iscrowd', False):
                    total_skipped_crowd += 1
                    continue
                if 'segmentation' not in ann or 'visible_mask' not in ann:
                    total_skipped_no_mask += 1
                    continue
                self.annotations.append(ann)
        
        print(
            f"✅ Hoàn tất! Gộp {len(ann_files)} files → {len(self.annotations)} mẫu"
            + (f" (bỏ {total_skipped_crowd} crowd)" if total_skipped_crowd else "")
            + (f" (bỏ {total_skipped_no_mask} thiếu mask)" if total_skipped_no_mask else "")
        )
