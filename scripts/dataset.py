"""
===================================================================================
AMODAL DATASET - Xử lý dữ liệu COCO-Amodal
===================================================================================
Dataset class để tải và xử lý dữ liệu amodal từ COCO annotation files.

Tính năng:
- Bóc tách từng vật thể riêng biệt (mỗi annotation object → 1 mẫu)
- Xây dựng mask "amodal" (hình dạng toàn bộ bao gồm phần bị che khuất)
- Xây dựng mask "visible" (chỉ phần nhìn thấy)
- Tính toán vùng bị che khuất (occlusion region)
- Tạo edge mask như gợi ý bổ sung
- Hỗ trợ data augmentation qua Albumentations
===================================================================================
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class AmodalDataset(Dataset):
    """
    Dataset class cho Amodal Shape Prediction trên COCO-Amodal.
    
    Cấu trúc dữ liệu COCO-Amodal:
    - Mỗi annotation có "regions" (danh sách vật thể)
    - Mỗi region có:
      - "segmentation": Đa giác tọa độ bao quanh vật thể (amodal shape)
      - "order": Độ sâu - vật thể nào phía trước (order nhỏ) che phía sau (order lớn)
    
    Xử lý:
    - Vẽ amodal mask (toàn bộ vật thể)
    - Vẽ visible mask bằng cách xóa bỏ phần bị che bởi các vật thể phía trước
    - Tính occlusion region = amodal - visible
    
    Args:
        img_dir: Đường dẫn thư mục chứa ảnh
        ann_file: Đường dẫn file annotation JSON (COCO format)
        transform: Hàm augmentation từ Albumentations (tùy chọn)
    """
    
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        print(f"📂 Đang nạp file annotation {ann_file} vào bộ nhớ...")
        # Khởi tạo COCO API
        self.coco = COCO(ann_file)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC QUAN TRỌNG: Bóc tách từng vật thể riêng biệt
        # ──────────────────────────────────────────────────────────────────
        # Thay vì 1 annotation = 1 mẫu, ở đây:
        # 1 annotation có thể có nhiều "regions" (vật thể)
        # → mỗi region = 1 mẫu huấn luyện
        self.instances = []

        for ann_id, ann in self.coco.anns.items():
            # Kiểm tra xem annotation có chứa "regions" (đặc tính amodal)
            if "regions" in ann:
                for region_idx, region in enumerate(ann["regions"]):
                    # Chỉ lấy những region có segmentation mask
                    if "segmentation" in region:
                        # Lưu cặp (annotation_id, chỉ số region trong annotation)
                        # Cặp này sẽ được lấy lại ở __getitem__
                        self.instances.append((ann_id, region_idx))

        print(
            f"✅ Hoàn tất! Đã bóc tách {len(self.instances)} mẫu vật thể riêng biệt."
        )

    def __len__(self):
        """Trả về tổng số mẫu trong dataset."""
        return len(self.instances)

    def __getitem__(self, idx):
        """
        Lấy một mẫu từ dataset.
        
        Quy trình:
        1. Lấy thông tin vật thể và ảnh từ annotation
        2. Đọc ảnh RGB từ file
        3. Vẽ amodal mask (toàn bộ vật thể)
        4. Vẽ visible mask (phần nhìn thấy sau khi xóa phần bị che)
        5. Tính occlusion region (phần bị che)
        6. Data augmentation (nếu có)
        7. Tính edge mask (viền gợi ý)
        8. Kết hợp thành 5 kênh input tensor
        
        Returns:
            Tuple gồm 4 thành phần:
            - input_tensor: Ảnh 5 kênh [5, H, W]
            - amodal_tensor: Amodal mask [H, W]
            - occluded_region: Vùng bị che [H, W]
            - cat_id: Class ID loại vật thể
        """
        
        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 1: LẤY THÔNG TIN VẬT THỂ & ẢNH
        # ──────────────────────────────────────────────────────────────────
        ann_id, region_idx = self.instances[idx]
        ann = self.coco.anns[ann_id]
        target_region = ann["regions"][region_idx]

        img_id = ann["image_id"]
        img_info = self.coco.loadImgs([img_id])[0]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 2: ĐỌC ẢNH RGB
        # ──────────────────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB
        height, width = image.shape[:2]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 3: VẼ AMODAL MASK (toàn bộ vật thể bao gồm phần bị che)
        # ──────────────────────────────────────────────────────────────────
        amodal_mask = np.zeros((height, width), dtype=np.uint8)
        segs = target_region["segmentation"]
        # Xử lý trường hợp segmentation được lưu dưới dạng list các đa giác
        if isinstance(segs, list) and len(segs) > 0:
            # Nếu là single polygon (list flat), chuyển thành list of polygons
            if isinstance(segs[0], (int, float)):
                segs = [segs]
            # Vẽ từng đa giác lên mask
            for poly in segs:
                if len(poly) >= 6:  # Cần ít nhất 3 điểm (6 tọa độ)
                    # Reshape từ list flat [x1, y1, x2, y2, ...] → [[x1,y1], [x2,y2], ...]
                    poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                    # Tô đầy đa giác bằng giá trị 1
                    cv2.fillPoly(amodal_mask, [poly_2d], 1)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 4: VẼ VISIBLE MASK (phần nhìn thấy)
        # ──────────────────────────────────────────────────────────────────
        # Bắt đầu bằng cách sao chép amodal mask
        visible_mask = amodal_mask.copy()
        # Lấy độ sâu (order) của vật thể hiện tại
        target_order = target_region.get("order", 0)

        # Xóa bỏ phần bị che bởi các vật thể phía trước (order < target_order)
        for other_region in ann["regions"]:
            other_order = other_region.get("order", 0)
            # Chỉ xem xét những vật thể phía trước (order nhỏ hơn)
            if other_order < target_order and "segmentation" in other_region:
                other_segs = other_region["segmentation"]
                if isinstance(other_segs, list) and len(other_segs) > 0:
                    if isinstance(other_segs[0], (int, float)):
                        other_segs = [other_segs]
                    # Vẽ mask của vật thể che khuất
                    for poly in other_segs:
                        if len(poly) >= 6:
                            poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                            # Tô đen (giá trị 0) để "xóa" phần bị che
                            cv2.fillPoly(visible_mask, [poly_2d], 0)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 5: DATA AUGMENTATION
        # ──────────────────────────────────────────────────────────────────
        if self.transform:
            # Áp dụng các phép biến đổi trên ảnh và cả 2 mask
            transformed = self.transform(image=image, masks=[amodal_mask, visible_mask])
            image = transformed["image"]
            amodal_mask = transformed["masks"][0]
            visible_mask = transformed["masks"][1]

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
        # Edge mask = biên của visible mask (dilatation - erosion)
        # Điều này giúp model biết ranh giới của vật thể nhìn thấy
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
        
        # Lấy class ID loại vật thể
        cat_id = ann.get('category_id', 0)

        return input_tensor, amodal_tensor, occluded_region, torch.tensor(cat_id, dtype=torch.long)