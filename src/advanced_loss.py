"""
Advanced Loss Functions & Sampling Strategy cho Amodal Occlusion Prediction.

Giải pháp cho vấn đề data imbalance:
1. FocalOcclusionLoss: Penalize hard negatives trong occluded regions (gamma=2)
2. WeightedOcclusionLoss: Hyperparameter-tuned weight (10x, 15x, 20x)
3. OcclusionFocalLoss: Combination of both
4. WeightedRandomSampler: Oversample occluded samples
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader


class FocalOcclusionLoss(nn.Module):
    """
    Focal Loss cho occlusion prediction.
    Hard negative mining: penalize model nếu sai đoán trên occluded region.

    Công thức: Focal Loss = -α * (1-p)^γ * log(p)
    - γ=2: tập trung vào hard negatives
    - α > 1: weighted occlusion loss
    """

    def __init__(self, alpha_occlusion=5.0, gamma=2.0):
        super().__init__()
        self.alpha_occlusion = alpha_occlusion
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, occluded_region):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] ground truth
            occluded_region: [B, 1, H, W] occlusion mask (0 or 1)
        """
        # BCE Loss
        bce_loss = self.bce(pred, target)  # [B, 1, H, W]

        # Probability of positive class
        pred_prob = torch.sigmoid(pred)

        # Focal loss term: (1 - p)^gamma for hard negatives
        # Khi model sai (p gần 0 nhưng target=1), (1-p) gần 1 → loss tăng
        focal_weight = torch.pow(1.0 - pred_prob.detach(), self.gamma)

        # Weighted BCE + Focal
        weighted_bce = bce_loss * focal_weight

        # Áp dụng occlusion weight
        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = self.alpha_occlusion

        weighted_focal_loss = (weighted_bce * weight_matrix).mean()

        # Dice loss
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_focal_loss + dice_loss.mean()


class OcclusionFocalLoss(nn.Module):
    """
    Combined approach: Focal Loss + Weighted Occlusion Loss.
    Tập trung vào:
    1. Hard negatives (focal)
    2. Occluded regions (weighted)
    """

    def __init__(self, alpha_occlusion=10.0, gamma=2.0, use_focal=True):
        super().__init__()
        self.alpha_occlusion = alpha_occlusion
        self.gamma = gamma
        self.use_focal = use_focal
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, occluded_region):
        bce_loss = self.bce(pred, target)
        pred_prob = torch.sigmoid(pred)

        # Focal weight nếu use_focal=True
        if self.use_focal:
            focal_weight = torch.pow(1.0 - pred_prob.detach(), self.gamma)
        else:
            focal_weight = 1.0

        # Occlusion weight
        weight_matrix = torch.ones_like(target)
        weight_matrix[occluded_region > 0.5] = self.alpha_occlusion

        # Combine focal + occlusion weighting
        weighted_bce = (bce_loss * focal_weight * weight_matrix).mean()

        # Dice Loss
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()


class OcclusionAwareLoss(nn.Module):
    """Original loss function (for reference & comparison)."""

    def __init__(self, occlusion_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
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
    Tạo WeightedRandomSampler để oversample mẫu occluded.

    Args:
        dataset: AmodalDataset object
        occlusion_threshold: Chỉ lấy samples có occlusion > threshold
        oversample_ratio: Bao nhiêu lần oversample (2.0 = gấp đôi)

    Returns:
        WeightedRandomSampler với weight cho mỗi sample
    """
    # Lặp qua dataset để tính occlusion ratio cho mỗi sample
    occlusion_ratios = []

    print(f"📊 Tính toán occlusion ratio cho {len(dataset)} samples...")
    for idx in range(len(dataset)):
        input_tensor, amodal_tensor, occluded_region, class_id = dataset[idx]

        # Tính occlusion ratio
        amodal_area = amodal_tensor.sum().item()
        occluded_area = occluded_region.sum().item()

        if amodal_area > 0:
            ratio = occluded_area / amodal_area
        else:
            ratio = 0.0

        occlusion_ratios.append(ratio)

    occlusion_ratios = np.array(occlusion_ratios)

    # Tạo weights dựa trên occlusion ratio
    weights = np.ones(len(dataset))

    # Oversample mẫu occluded
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
    Tạo DataLoader với weighted sampling strategy.

    Args:
        dataset: AmodalDataset
        batch_size: Batch size
        num_workers: Số workers
        occlusion_threshold: Threshold để xác định "occluded" sample
        oversample_ratio: Bao nhiêu lần oversample
        use_weighted_sampler: Có sử dụng WeightedSampler hay không

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
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader


# Test functions
if __name__ == "__main__":
    print("🧪 Test Loss Functions\n")

    # Dummy data
    B, H, W = 2, 224, 224
    pred = torch.randn(B, 1, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)
    occluded = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)

    # Test Original Loss
    print("1️⃣ OcclusionAwareLoss (5x)")
    loss_5x = OcclusionAwareLoss(occlusion_weight=5.0)
    loss_val = loss_5x(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    # Test FocalOcclusionLoss
    print("2️⃣ FocalOcclusionLoss (alpha=5, gamma=2)")
    focal_loss = FocalOcclusionLoss(alpha_occlusion=5.0, gamma=2.0)
    loss_val = focal_loss(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    # Test OcclusionFocalLoss with 10x weight
    print("3️⃣ OcclusionFocalLoss (10x weight, gamma=2)")
    combo_loss = OcclusionFocalLoss(alpha_occlusion=10.0, gamma=2.0, use_focal=True)
    loss_val = combo_loss(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    # Test OcclusionFocalLoss with 15x weight
    print("4️⃣ OcclusionFocalLoss (15x weight, gamma=2)")
    combo_loss = OcclusionFocalLoss(alpha_occlusion=15.0, gamma=2.0, use_focal=True)
    loss_val = combo_loss(pred, target, occluded)
    print(f"   Loss: {loss_val.item():.4f}\n")

    print("✅ All loss functions working correctly!")
