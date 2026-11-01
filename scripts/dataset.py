"""
===================================================================================
AMODAL DATASET - Xử lý dữ liệu COCO-Amodal
===================================================================================
Dataset class để tải và xử lý dữ liệu amodal từ COCO annotation files.

Tính năng:
- Bóc tách từng vật thể riêng biệt (mỗi annotation object → 1 mẫu)
- Trích xuất class_id đa tầng (category_id -> name tra bảng COCO 80/91 -> CRC32 -> parent_ann)
- Xây dựng mask "amodal" (hình dạng toàn bộ bao gồm phần bị che khuất)
- Xây dựng mask "visible" (chỉ phần nhìn thấy)
- Tính toán vùng bị che khuất (occlusion region)
- Tạo edge mask như gợi ý bổ sung
- Hỗ trợ data augmentation qua Albumentations
===================================================================================
"""

import os
import zlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


# Bảng ánh xạ nhãn tên COCO chuẩn 80/91 lớp
COCO_NAME_TO_ID = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "street sign": 12, "stop sign": 13, "parking meter": 14, "bench": 15,
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20,
    "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25,
    "hat": 26, "backpack": 27, "umbrella": 28, "shoe": 29, "eye glasses": 30,
    "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34, "skis": 35,
    "snowboard": 36, "sports ball": 37, "kite": 38, "baseball bat": 39, "baseball glove": 40,
    "skateboard": 41, "surfboard": 42, "tennis racket": 43, "bottle": 44, "plate": 45,
    "wine glass": 46, "cup": 47, "fork": 48, "knife": 49, "spoon": 50,
    "bowl": 51, "banana": 52, "apple": 53, "sandwich": 54, "orange": 55,
    "broccoli": 56, "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60,
    "cake": 61, "chair": 62, "couch": 63, "potted plant": 64, "bed": 65,
    "mirror": 66, "dining table": 67, "window": 68, "desk": 69, "toilet": 70,
    "door": 71, "tv": 72, "laptop": 73, "mouse": 74, "remote": 75,
    "keyboard": 76, "cell phone": 77, "microwave": 78, "oven": 79, "toaster": 80,
    "sink": 81, "refrigerator": 82, "blender": 83, "book": 84, "clock": 85,
    "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89, "toothbrush": 90
}


def extract_cocoa_category_id(target_region: dict, parent_ann: dict = None, num_classes: int = 91) -> int:
    """
    Trích xuất category_id chính xác cho từng region của COCOA trong khoảng [0, num_classes - 1].
    Ưu tiên:
      1. target_region['category_id'] (nếu là số nguyên hợp lệ)
      2. target_region['name'] (tra bảng chuẩn COCO_NAME_TO_ID, nếu là nhãn mở rộng của COCOA
         thì dùng băm tất định CRC32 vào khoảng [1, num_classes - 1])
      3. parent_ann['category_id'] (nếu có)
      4. Fallback về 0
    """
    if parent_ann is None:
        parent_ann = {}

    # 1. Kiểm tra category_id trực tiếp trong target_region
    if isinstance(target_region, dict) and "category_id" in target_region and target_region["category_id"] is not None:
        try:
            cid = int(target_region["category_id"])
            if cid > 0:
                return cid % num_classes
        except (ValueError, TypeError):
            pass

    # 2. Kiểm tra trường chuỗi văn bản 'name' trong target_region (chuẩn Yan Zhu CVPR 2017)
    if isinstance(target_region, dict):
        region_name = target_region.get("name", None)
        if isinstance(region_name, str) and len(region_name.strip()) > 0:
            clean_name = region_name.strip().lower()
            if clean_name in COCO_NAME_TO_ID:
                return COCO_NAME_TO_ID[clean_name] % num_classes
            # Với các nhãn open-vocabulary / stuff trong COCOA (ví dụ: 'grass', 'wall', 'table'...)
            # băm cố định tất định về [1, num_classes - 1]
            return 1 + (zlib.crc32(clean_name.encode("utf-8")) % (num_classes - 1))

    # 3. Fallback sang parent_ann
    if isinstance(parent_ann, dict) and "category_id" in parent_ann and parent_ann["category_id"] is not None:
        try:
            cid = int(parent_ann["category_id"])
            if cid > 0:
                return cid % num_classes
        except (ValueError, TypeError):
            pass

    return 0


class AmodalDataset(Dataset):
    """
    Dataset class cho Amodal Shape Prediction trên COCO-Amodal.
    
    Cấu trúc dữ liệu COCO-Amodal:
    - Mỗi annotation có "regions" (danh sách vật thể)
    - Mỗi region có:
      - "segmentation": Đa giác tọa độ bao quanh vật thể (amodal shape)
      - "order": Độ sâu - vật thể nào phía trước (order nhỏ) che phía sau (order lớn)
      - "category_id" hoặc "name": Định danh hoặc tên lớp đối tượng
    
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
            f"✅ Hoàn tất! Đã bóc tách {len(self.instances):,} mẫu vật thể riêng biệt."
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
        9. Trích xuất class ID qua extract_cocoa_category_id
        
        Returns:
            Tuple gồm 4 thành phần:
            - input_tensor: Ảnh 5 kênh [5, H, W]
            - amodal_tensor: Amodal mask [H, W]
            - occluded_region: Vùng bị che [H, W]
            - cat_id: Class ID loại vật thể (Tensor scalar long)
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
        if image is None:
            # Fallback nếu định dạng đuôi file khác nhau
            alt_path = img_path.replace(".jpg", ".png") if img_path.endswith(".jpg") else img_path.replace(".png", ".jpg")
            image = cv2.imread(alt_path)

        if image is None:
            h = img_info.get("height", 224)
            w = img_info.get("width", 224)
            image = np.zeros((h, w, 3), dtype=np.uint8)
        else:
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
                    poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(amodal_mask, [poly_2d], 1)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 4: VẼ VISIBLE MASK (phần nhìn thấy)
        # ──────────────────────────────────────────────────────────────────
        visible_mask = amodal_mask.copy()
        target_order = target_region.get("order", 0)

        # Xóa bỏ phần bị che bởi các vật thể phía trước (order < target_order)
        for other_region in ann["regions"]:
            other_order = other_region.get("order", 0)
            if other_order < target_order and "segmentation" in other_region:
                other_segs = other_region["segmentation"]
                if isinstance(other_segs, list) and len(other_segs) > 0:
                    if isinstance(other_segs[0], (int, float)):
                        other_segs = [other_segs]
                    for poly in other_segs:
                        if len(poly) >= 6:
                            poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(visible_mask, [poly_2d], 0)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 5: DATA AUGMENTATION
        # ──────────────────────────────────────────────────────────────────
        if self.transform:
            transformed = self.transform(image=image, masks=[amodal_mask, visible_mask])
            image = transformed["image"]
            amodal_mask = transformed["masks"][0]
            visible_mask = transformed["masks"][1]

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 6: CHUYỂN ĐỔI TỪ NUMPY → PYTORCH TENSOR
        # ──────────────────────────────────────────────────────────────────
        if not isinstance(image, torch.Tensor):
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            image_tensor = image

        amodal_tensor = torch.from_numpy(amodal_mask).float()
        visible_tensor = torch.from_numpy(visible_mask).float()

        # Tính vùng bị che khuất: amodal - visible (bằng 0 nếu không bị che)
        occluded_region = torch.clamp(amodal_tensor - visible_tensor, min=0.0)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 7: VẼ EDGE MASK (Viền gợi ý)
        # ──────────────────────────────────────────────────────────────────
        visible_uint8 = (visible_mask * 255).astype(np.uint8) 
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(visible_uint8, kernel, iterations=1)
        erosion = cv2.erode(visible_uint8, kernel, iterations=1)
        edge_mask = torch.tensor((dilation - erosion) / 255.0, dtype=torch.float32)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 8: KẾT HỢP THÀNH 5 KÊNH INPUT
        # ──────────────────────────────────────────────────────────────────
        input_tensor = torch.cat([
            image_tensor,                          # Kênh 0-2: RGB [3, H, W]
            visible_tensor.unsqueeze(0),           # Kênh 3: Visible [1, H, W]
            edge_mask.unsqueeze(0)                 # Kênh 4: Edge [1, H, W]
        ], dim=0)

        # ──────────────────────────────────────────────────────────────────
        # BƯỚC 9: TRÍCH XUẤT CLASS ID CHÍNH XÁC (ĐA TẦNG CHO COCOA)
        # ──────────────────────────────────────────────────────────────────
        cat_id = extract_cocoa_category_id(target_region, ann, num_classes=91)

        return input_tensor, amodal_tensor, occluded_region, torch.tensor(cat_id, dtype=torch.long)
