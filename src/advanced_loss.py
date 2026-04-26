"""
===================================================================================
ADVANCED LOSS FUNCTIONS & SAMPLING STRATEGIES
===================================================================================
Các hàm loss nâng cao và chiến lược sampling để xử lý data imbalance trong occlusion.

Giải pháp cho vấn đề:
- Class imbalance: Vùng occlusion chiếm ít pixel hơn vùng không occlusion
- Hard negatives: Mô hình sai đoán nhiều ở vùng khó

Các loss function:
1. FocalOcclusionLoss: Focal Loss cơ bản cho occlusion
2. OcclusionFocalLoss: Kết hợp Focal + Weighted Occlusion
3. OcclusionAwareLoss: Loss gốc (tham chiếu)

Các sampling strategy:
1. WeightedRandomSampler: Oversample mẫu có occlusion
2. create_balanced_dataloader: Tạo DataLoader với balanced sampling

===================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader


class FocalOcclusionLoss(nn.Module):
    """
    Focal Loss thiết kế cho occlusion prediction.
    
    Ý tưởng:
    - Focal Loss tập trung vào hard negatives (những trường hợp model sai đoán)
    - Công thức: Loss = -α(1-p_t)^γ * log(p_t)
      + γ=2: tăng trọng lượng cho hard negatives
      + α=5: thêm weight cho occlusion regions
    
    Lợi ích:
    - Giải quyết class imbalance tự nhiên
    - Không cần oversampling phức tạp
    - Hiệu suất tốt với dataset ít dữ liệu
    
    Args:
        alpha_occlusion: Hệ số nhân cho vùng occlusion (mặc định: 5.0)
        gamma: Exponent cho focal term (mặc định: 2.0)
    """

    def __init__(self, alpha_occlusion=5.0, gamma=2.0):
        super().__init__()
        self.alpha_occlusion = alpha_occlusion
        self.gamma = gamma
        # BCE loss tính từng pixel riêng (không lấy mean ngay)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, occluded_region):
        """
        Tính Focal loss với weight cho occlusion region.
        
        Args:
            pred: Dự đoán logit [B, 1, H, W]
            target: Amodal mask nhãn [B, 1, H, W]
            occluded_region: Vùng occlusion [B, 1, H, W]
        
        Returns:
            Tổng loss (scalar)
        """
        # Tính BCE loss cho từng pixel
        bce_loss = self.bce(pred, target)  # [B, 1, H, W]

        # Chuyển logit thành probability
        pred_prob = torch.sigmoid(pred)

        # Tính Focal weight: (1 - p)^γ
        # Khi model sai (p gần 0 nhưng target=1): (1-p)→1 → loss tăng
        # Khi model đúng (p gần 1): (1-p)→0 → loss giảm
        focal_weight = torch.pow(1.0 - pred_prob.detach(), self.gamma)

        # Kết hợp focal weight vào BCE
        weighted_bce = bce_loss * focal_weight

        # Thêm weight cho occlusion region
        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = self.alpha_occlusion

        weighted_focal_loss = (weighted_bce * weight_matrix).mean()

        # Kết hợp Dice loss để cân bằng
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_focal_loss + dice_loss.mean()


class OcclusionFocalLoss(nn.Module):
    """
    Kết hợp Focal Loss + Weighted Occlusion Loss.
    
    Tập trung vào 2 vấn đề:
    1. Hard negatives (focal)
    2. Occlusion regions (weighted)
    
    Args:
        alpha_occlusion: Hệ số nhân cho occlusion (mặc định: 10.0)
        gamma: Exponent focal (mặc định: 2.0)
        use_focal: Có dùng focal weighting không (mặc định: True)
    """

    def __init__(self, alpha_occlusion=10.0, gamma=2.0, use_focal=True):
        super().__init__()
        self.alpha_occlusion = alpha_occlusion
        self.gamma = gamma
        self.use_focal = use_focal
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, occluded_region):
        """
        Tính combined loss: focal + weighted occlusion.
        """
        bce_loss = self.bce(pred, target)
        pred_prob = torch.sigmoid(pred)

        # Focal weight (nếu sử dụng)
        if self.use_focal:
            focal_weight = torch.pow(1.0 - pred_prob.detach(), self.gamma)
        else:
            focal_weight = 1.0

        # Occlusion weight
        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = self.alpha_occlusion

        # Kết hợp: focal * occlusion
        weighted_bce = (bce_loss * focal_weight * weight_matrix).mean()

        # Dice Loss
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()


class OcclusionAwareLoss(nn.Module):
    """
    Loss function gốc (tham chiếu để so sánh).
    
    Cấu trúc:
    - Weighted BCE: Weight cao ở occlusion, thấp ở non-occlusion
    - Dice Loss: Cân bằng scale
    
    Args:
        occlusion_weight: Hệ số nhân cho occlusion (mặc định: 5.0)
    """

    def __init__(self, occlusion_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        """Tính weighted BCE + Dice loss."""
        bce_loss = self.bce(pred, target)
        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = self.occlusion_weight
        weighted_bce = (bce_loss * weight_matrix).mean()

        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()


def create_occluded_sampler(dataset, occlusion_threshold=0.1, oversample_ratio=2.0):
    """
    Tạo WeightedRandomSampler để oversample mẫu có occlusion.
    
    Chiến lược:
    - Duyệt qua dataset tính occlusion ratio
    - Tạo weight cao cho sample có occlusion
    - Sampling với replacement để balance dataset
    
    Args:
        dataset: AmodalDataset object
        occlusion_threshold: Threshold occlusion ratio (mặc định: 0.1 = 10%)
        oversample_ratio: Bao nhiêu lần oversample (mặc định: 2.0 = 2x)
    
    Returns:
        Tuple:
        - sampler: WeightedRandomSampler
        - occlusion_ratios: Array occlusion ratio của mỗi sample
    """
    # Duyệt dataset để tính occlusion ratio
    occlusion_ratios = []

    print(f"📊 Tính occlusion ratio cho {len(dataset)} mẫu...")
    for idx in range(len(dataset)):
        input_tensor, amodal_tensor, occluded_region, class_id = dataset[idx]

        # Tính tỷ lệ occlusion = occluded_area / amodal_area
        amodal_area = amodal_tensor.sum().item()
        occluded_area = occluded_region.sum().item()

        if amodal_area > 0:
            ratio = occluded_area / amodal_area
        else:
            ratio = 0.0

        occlusion_ratios.append(ratio)

    occlusion_ratios = np.array(occlusion_ratios)

    # Tạo weight dựa trên occlusion ratio
    weights = np.ones(len(dataset))

    # Oversample mẫu có occlusion cao
    occluded_mask = occlusion_ratios > occlusion_threshold
    weights[occluded_mask] = oversample_ratio

    occluded_count = occluded_mask.sum()
    print(
        f"✅ Tìm thấy {occluded_count} mẫu có occlusion > {100*occlusion_threshold:.0f}%"
    )
    print(f"📈 Oversample ratio: {oversample_ratio}x cho mẫu occluded")

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True
    )

    return sampler, occlusion_ratios


def create_balanced_dataloader(
    dataset,
    batch_size=4,
    num_workers=0,
    occlusion_threshold=0.1,
    oversample_ratio=2.0,
    use_weighted_sampler=True,
):
    """
    Tạo DataLoader với balanced sampling strategy.
    
    Args:
        dataset: AmodalDataset
        batch_size: Kích thước batch
        num_workers: Số workers cho parallelization
        occlusion_threshold: Threshold occlusion (10%)
        oversample_ratio: Oversample multiplier (2x)
        use_weighted_sampler: Có dùng WeightedSampler không
    
    Returns:
        DataLoader với balanced sampling
    """
    if use_weighted_sampler:
        sampler, ratios = create_occluded_sampler(
            dataset,
            occlusion_threshold=occlusion_threshold,
            oversample_ratio=oversample_ratio,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Random sampling (không balanced)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader


# ===================================================================================
# PHẦN TEST - Kiểm tra các loss functions
# ===================================================================================
if __name__ == "__main__":
    print("🧪 Test Loss Functions\n")

    # Dữ liệu giả định
    B, H, W = 2, 224, 224
    pred = torch.randn(B, 1, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)
    occluded = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)

    # Test 1: OcclusionAwareLoss (5x)
    print("1️⃣ OcclusionAwareLoss (5x weight)")
    loss_5x = OcclusionAwareLoss(occlusion_weight=5.0)
    loss_val = loss_5x(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    # Test 2: FocalOcclusionLoss
    print("2️⃣ FocalOcclusionLoss (alpha=5, gamma=2)")
    focal_loss = FocalOcclusionLoss(alpha_occlusion=5.0, gamma=2.0)
    loss_val = focal_loss(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    # Test 3: OcclusionFocalLoss (10x)
    print("3️⃣ OcclusionFocalLoss (10x weight, gamma=2)")
    combo_loss = OcclusionFocalLoss(alpha_occlusion=10.0, gamma=2.0, use_focal=True)
    loss_val = combo_loss(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    # Test 4: OcclusionFocalLoss (15x)
    print("4️⃣ OcclusionFocalLoss (15x weight, gamma=2)")
    combo_loss = OcclusionFocalLoss(alpha_occlusion=15.0, gamma=2.0, use_focal=True)
    loss_val = combo_loss(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    print("✅ Tất cả loss functions đều hoạt động bình thường!")
