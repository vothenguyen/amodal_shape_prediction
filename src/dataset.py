import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class AmodalDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        print(f"Đang nạp file nhãn {ann_file} vào bộ nhớ...")
        self.coco = COCO(ann_file)

        # SỨC MẠNH LÀ Ở ĐÂY: Bóc tách từng vật thể độc lập
        self.instances = []

        for ann_id, ann in self.coco.anns.items():
            if "regions" in ann:
                for region_idx, region in enumerate(ann["regions"]):
                    # Chỉ lấy những vùng thực sự có chứa tọa độ mask
                    if "segmentation" in region:
                        # Lưu lại cặp (ID Ảnh, Vị trí vật thể trong ảnh)
                        self.instances.append((ann_id, region_idx))

        print(
            f"Hoàn tất! Đã bóc tách thành công {len(self.instances)} vật thể độc lập."
        )

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # 1. Lấy thông tin vật thể (Target) và Ảnh
        ann_id, region_idx = self.instances[idx]
        ann = self.coco.anns[ann_id]
        target_region = ann["regions"][region_idx]

        img_id = ann["image_id"]
        img_info = self.coco.loadImgs([img_id])[0]

        # 2. Đọc ảnh RGB
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # 3. Vẽ AMODAL MASK (Dùng làm Nhãn / Label để chấm điểm)
        amodal_mask = np.zeros((height, width), dtype=np.uint8)
        segs = target_region["segmentation"]
        if isinstance(segs, list) and len(segs) > 0:
            if isinstance(segs[0], (int, float)):
                segs = [segs]
            for poly in segs:
                if len(poly) >= 6:
                    poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(amodal_mask, [poly_2d], 1)

        # 4. TẠO VISIBLE MASK (Kênh thứ 4) BẰNG CÁCH "GỌT" MASK
        # Copy từ Amodal Mask làm gốc
        visible_mask = amodal_mask.copy()
        target_order = target_region.get("order", 0)

        # Duyệt qua các vật thể khác CÙNG BỨC ẢNH
        for other_region in ann["regions"]:
            other_order = other_region.get("order", 0)

            # Giả định: order NHỎ HƠN nghĩa là nằm TRƯỚC (đè lên vật thể của mình)
            if other_order < target_order and "segmentation" in other_region:
                other_segs = other_region["segmentation"]
                if isinstance(other_segs, list) and len(other_segs) > 0:
                    if isinstance(other_segs[0], (int, float)):
                        other_segs = [other_segs]
                    for poly in other_segs:
                        if len(poly) >= 6:
                            poly_2d = np.array(poly).reshape(-1, 2).astype(np.int32)
                            # Tô màu đen (0) đè lên để XÓA phần bị che khuất
                            cv2.fillPoly(visible_mask, [poly_2d], 0)

        # 5. Đưa qua hàm Resize (về 224x224)
        if self.transform:
            # Truyền cùng lúc cả 2 mask vào để cắt ghép cho đồng bộ
            transformed = self.transform(image=image, masks=[amodal_mask, visible_mask])
            image = transformed["image"]
            amodal_mask = transformed["masks"][0]
            visible_mask = transformed["masks"][1]

        # 6. ÉP KIỂU VÀ GHÉP THÀNH TENSOR 4 KÊNH
        # Ảnh RGB: [H, W, 3] -> [3, H, W]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Visible Mask: [H, W] -> Thêm chiều thành [1, H, W]
        visible_tensor = torch.from_numpy(visible_mask).float().unsqueeze(0)

        # SIÊU HỢP THỂ: Nối 3 kênh RGB và 1 kênh Visible lại -> [4, H, W]
        input_4channel = torch.cat([image_tensor, visible_tensor], dim=0)

        # Amodal Mask (Nhãn dán): [H, W] (Giữ nguyên dùng để tính loss)
        amodal_target = torch.from_numpy(amodal_mask).long()

        # Nhả ra cho PyTorch: (Input 4 Kênh, Output 1 Kênh)
        occluded_region = torch.clamp(amodal_mask - visible_mask, min=0.0)

        # 2. KÊNH THỨ 5: Tìm "Viền đứt gãy" (Edge Mask) bằng Toán học hình thái (Morphology)
        visible_np = visible_mask.numpy()
        kernel = np.ones((5, 5), np.uint8)  # Dùng ma trận 5x5 để làm viền dày lên xíu
        dilation = cv2.dilate(visible_np, kernel, iterations=1)
        erosion = cv2.erode(visible_np, kernel, iterations=1)
        edge_mask = torch.tensor(dilation - erosion, dtype=torch.float32)

        # 3. Ghép thành Input 5 Kênh (3 màu + 1 Visible + 1 Viền đứt gãy)
        input_tensor = torch.cat(
            [img_tensor, visible_mask.unsqueeze(0), edge_mask.unsqueeze(0)], dim=0
        )

        # Trả về 3 món: Ảnh 5 kênh, Đáp án Amodal, và Vùng bị khuất (để tính điểm)
        return input_tensor, amodal_mask, occluded_region
