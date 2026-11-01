"""
===================================================================================
HUẤN LUYỆN AMODAL SWIN-UNET (ROW 5 D2SA)
===================================================================================
Cấu hình: Row 5 Ablation Study:
- Đầu vào: 5 kênh (RGB + Visible Mask + Edge Mask)
- Category Embedding: Tắt
- Spatial Attention: Bật (kernel=7)
- Hàm mất mát: OcclusionAwareLoss (5x trọng số vùng che) + Dice Loss
- Dữ liệu: D2SA (training_rot0 + augmented)

Chạy: python scripts/other_config_d2sa/amodal_shape_prediction_no_embeding/train.py
===================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Đảm bảo import đúng model.py cục bộ trước, sau đó mới nạp dataset_d2sa từ scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from dataset_d2sa import AmodalDatasetD2SA_Concat, AmodalDatasetD2SA
from model import AmodalSwinUNet
from logging_utils import TrainLogger


class OcclusionAwareLoss(nn.Module):
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


def calculate_val_metrics(pred_mask, gt_amodal, visible_mask):
    """
    Tính toán các chỉ số đánh giá cho 1 sample:
    - mIoU (IoU toàn thể amodal mask)
    - Dice
    - Precision
    - Recall
    - Invisible mIoU (IoU chỉ tính riêng trên vùng bị che khuất: invisible = amodal & ~visible)
    """
    intersection = np.logical_and(pred_mask, gt_amodal).sum()
    union = np.logical_or(pred_mask, gt_amodal).sum()
    iou = intersection / union if union > 0 else 0.0

    pred_sum = pred_mask.sum()
    gt_sum = gt_amodal.sum()
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0

    inv_gt = np.logical_and(gt_amodal, np.logical_not(visible_mask))
    inv_pred = np.logical_and(pred_mask, np.logical_not(visible_mask))
    has_occlusion = inv_gt.sum() > 0

    if has_occlusion:
        inv_intersection = np.logical_and(inv_pred, inv_gt).sum()
        inv_union = np.logical_or(inv_pred, inv_gt).sum()
        invisible_iou = inv_intersection / inv_union if inv_union > 0 else 0.0
    else:
        invisible_iou = None

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "invisible_iou": float(invisible_iou) if invisible_iou is not None else 0.0,
        "has_occlusion": bool(has_occlusion)
    }


def evaluate_validation(model, val_loader, device, threshold=0.5):
    """Chạy đánh giá trên DataLoader validation và trả về dict các chỉ số trung bình."""
    model.eval()
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "inv_iou": 0.0}
    occluded_count = 0
    total_count = 0
    total_pos_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, targets, _, _ in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs)
            preds_binary = (probs > threshold).cpu().numpy().astype(np.uint8)
            targets_np = targets.unsqueeze(1).numpy().astype(np.uint8)
            visible_np = inputs[:, 3:4, :, :].cpu().numpy().astype(np.uint8)

            B = inputs.size(0)
            for b in range(B):
                res = calculate_val_metrics(
                    preds_binary[b, 0], targets_np[b, 0], visible_np[b, 0]
                )
                total_pos_pixels += int((preds_binary[b, 0] == 1).sum())
                total_pixels += preds_binary[b, 0].size
                total_metrics["iou"] += res["iou"]
                total_metrics["dice"] += res["dice"]
                total_metrics["precision"] += res["precision"]
                total_metrics["recall"] += res["recall"]
                if res["has_occlusion"]:
                    total_metrics["inv_iou"] += res["invisible_iou"]
                    occluded_count += 1
                total_count += 1

    if total_count == 0:
        return {}

    val_pos_pct = round(float(total_pos_pixels / total_pixels * 100.0), 2) if total_pixels > 0 else 0.0

    return {
        "mIoU": round(total_metrics["iou"] / total_count, 4),
        "invisible_mIoU": round(total_metrics["inv_iou"] / occluded_count, 4) if occluded_count > 0 else 0.0,
        "precision": round(total_metrics["precision"] / total_count, 4),
        "recall": round(total_metrics["recall"] / total_count, 4),
        "dice": round(total_metrics["dice"] / total_count, 4),
        "pos_pixel_pct": val_pos_pct,
    }


def train(args):
    args.checkpoint_dir = os.path.normpath(args.checkpoint_dir)
    args.log_dir = os.path.normpath(args.log_dir)
    if hasattr(args, "img_dir") and args.img_dir:
        args.img_dir = os.path.normpath(args.img_dir)
    if hasattr(args, "ann_files") and args.ann_files:
        args.ann_files = [os.path.normpath(p) for p in args.ann_files]
    if hasattr(args, "val_ann_file") and args.val_ann_file:
        args.val_ann_file = os.path.normpath(args.val_ann_file)
    if hasattr(args, "resume_checkpoint") and args.resume_checkpoint:
        args.resume_checkpoint = os.path.normpath(args.resume_checkpoint)

    logger = TrainLogger(
        config_name="Row 5 (5ch, OccLoss 5x, Edge, No Emb, Spatial)",
        log_dir=args.log_dir
    )
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ─────────────────────────────────────────────────────────────────
    # KHỞI TẠO MÔ HÌNH & KIỂM CHỨNG KIẾN TRÚC / THAM SỐ
    # ─────────────────────────────────────────────────────────────────
    EXPECTED_PARAMS = 34_353_661  # Row 5 (5ch, OccLoss 5x, Edge, No Emb, Spatial)
    model = AmodalSwinUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params == EXPECTED_PARAMS, (
        f"LỖI KIẾN TRÚC MÔ HÌNH (Row 5): Tổng tham số thực tế là {total_params:,}, "
        f"nhưng kỳ vọng chính xác là {EXPECTED_PARAMS:,}! "
        f"(Kiểm tra lại số kênh patch_embed=5, không category embedding, có spatial attention kernel 7)."
    )

    logger.log_start(
        args,
        extra_info={
            "Tổng tham số": f"{total_params:,}",
            "Kỳ vọng": f"{EXPECTED_PARAMS:,}",
            "Trainable": f"{trainable_params:,}",
            "Kiểm chứng kiến trúc": "KHỚP 100% (ASSERTION PASSED)"
        }
    )
    logger.info(f"Thiết bị huấn luyện: {device}")

    # ─────────────────────────────────────────────────────────────────
    # LOSS FUNCTION & OPTIMIZER
    # ─────────────────────────────────────────────────────────────────
    criterion = OcclusionAwareLoss(occlusion_weight=args.occlusion_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Đăng ký signal handler để lưu khẩn cấp (gồm model, optimizer, scheduler) khi nhận SIGTERM
    logger.register_signal_handler(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        optimizer=optimizer,
        scheduler=scheduler
    )

    train_transform = A.Compose([
        A.Resize(args.resize, args.resize),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    logger.info("Chuẩn bị D2SA DataLoader (training_rot0 + augmented)...")
    if len(args.ann_files) == 1:
        train_dataset = AmodalDatasetD2SA(
            img_dir=args.img_dir, ann_file=args.ann_files[0], transform=train_transform
        )
    else:
        train_dataset = AmodalDatasetD2SA_Concat(
            img_dir=args.img_dir, ann_files=args.ann_files, transform=train_transform
        )

    if args.subset_size and args.subset_size < len(train_dataset):
        logger.info(f"Chế độ Subset: Chỉ lấy {args.subset_size}/{len(train_dataset)} mẫu!")
        train_dataset.annotations = train_dataset.annotations[:args.subset_size]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Khởi tạo Validation DataLoader
    val_loader = None
    final_val_loader = None
    if args.val_ann_file and os.path.exists(args.val_ann_file) and (args.eval_final or args.eval_every > 0):
        val_transform = A.Resize(args.resize, args.resize)
        full_val_dataset = AmodalDatasetD2SA(
            img_dir=args.img_dir,
            ann_file=args.val_ann_file,
            transform=val_transform
        )
        final_val_loader = DataLoader(
            full_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )
        if args.val_subset_size and args.val_subset_size < len(full_val_dataset):
            import copy
            inter_val_dataset = copy.copy(full_val_dataset)
            inter_val_dataset.annotations = full_val_dataset.annotations[:args.val_subset_size]
            val_loader = DataLoader(
                inter_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True if device.type == "cuda" else False,
            )
            logger.info(f"Đã chuẩn bị validation: Định kỳ ({len(val_loader.dataset)} mẫu) | Cuối kỳ ({len(final_val_loader.dataset)} mẫu).")
        else:
            val_loader = final_val_loader
            logger.info(f"Đã nạp {len(full_val_dataset)} mẫu validation để đánh giá định kỳ / tổng kết.")

        # Resume checkpoint nếu được yêu cầu
    start_epoch = args.resume_epoch
    weight_path = None
    regular_path = os.path.join(args.checkpoint_dir, f"swin_amodal_epoch_{start_epoch}.pth") if start_epoch > 0 else None
    emergency_path = os.path.join(args.checkpoint_dir, f"emergency_checkpoint_epoch_{start_epoch}.pth") if start_epoch > 0 else None
    last_path = os.path.join(args.checkpoint_dir, "last.pth")

    if getattr(args, "resume_checkpoint", None) and os.path.exists(args.resume_checkpoint):
        weight_path = args.resume_checkpoint
        logger.info(f"Chỉ định checkpoint nạp trực tiếp: {weight_path}")
    elif start_epoch > 0:
        if emergency_path and os.path.exists(emergency_path):
            weight_path = emergency_path
            logger.info(f"Phát hiện checkpoint khẩn cấp tại: {emergency_path}")
        elif os.path.exists(last_path):
            try:
                ckpt_meta = torch.load(last_path, map_location="cpu")
                if isinstance(ckpt_meta, dict) and ckpt_meta.get("epoch") == start_epoch:
                    weight_path = last_path
                    logger.info(f"Phát hiện last.pth đầy đủ trạng thái khớp Epoch {start_epoch}: {last_path}")
                else:
                    found_ep = ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else "không rõ"
                    logger.warning(f"last.pth chứa epoch {found_ep} != start_epoch {start_epoch}. Sẽ thử regular checkpoint.")
            except Exception as e_last:
                logger.warning(f"[CẢNH BÁO] Không thể đọc last.pth ({e_last}). Tự động rơi về fallback {regular_path}!")

        if weight_path is None and regular_path and os.path.exists(regular_path):
            weight_path = regular_path
            logger.info(f"Sử dụng checkpoint định kỳ: {regular_path}")
        elif weight_path is None and not (emergency_path and os.path.exists(emergency_path)):
            logger.warning(f"Không tìm thấy checkpoint tại {regular_path}, {emergency_path} hoặc {last_path}. Bắt đầu từ Epoch 0!")
            start_epoch = 0

    has_resumed_scheduler = False
    if weight_path and os.path.exists(weight_path):
        ckpt = None
        try:
            ckpt = torch.load(weight_path, map_location=device)
        except Exception as load_err:
            logger.warning(f"[CẢNH BÁO] Lỗi khi nạp checkpoint từ {weight_path}: {load_err}")
            if regular_path and os.path.exists(regular_path) and weight_path != regular_path:
                logger.warning(f"Rơi về checkpoint định kỳ dự phòng: {regular_path}")
                try:
                    ckpt = torch.load(regular_path, map_location=device)
                    weight_path = regular_path
                except Exception as reg_err:
                    logger.error(f"[LỖI] Fallback về regular checkpoint cũng thất bại: {reg_err}")
                    ckpt = None
            else:
                ckpt = None

        if ckpt is not None:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                if 'optimizer_state_dict' in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        logger.info("Đã phục hồi hoàn toàn trạng thái optimizer (AdamW moments) từ checkpoint!")
                    except Exception as opt_err:
                        logger.warning(f"Không thể nạp optimizer state: {opt_err}")
                if 'scheduler_state_dict' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                        has_resumed_scheduler = True
                        logger.info(f"Đã phục hồi scheduler state: LR = {scheduler.get_last_lr()[0]:.6e}")
                    except Exception as sched_err:
                        logger.warning(f"Không thể nạp scheduler state: {sched_err}")
            else:
                state_dict = ckpt
            cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned)
            logger.info(f"Tiếp tục từ Epoch {start_epoch}: Đã nạp thành công weights tại {weight_path}")
        else:
            logger.warning(f"Không thể nạp weights từ checkpoint. Model sẽ dùng weights ban đầu!")

    if start_epoch > 0 and not has_resumed_scheduler:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()
        logger.info(f"Đã đồng bộ learning rate schedule về Epoch {start_epoch} (fast-forward): LR = {scheduler.get_last_lr()[0]:.6e}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_metrics = None

    try:
        logger.info(f"BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN: Epoch {start_epoch + 1} -> {args.epochs}")

        for epoch in range(start_epoch, args.epochs):
            logger.set_current_epoch(epoch + 1)
            t_epoch_start = time.time()
            model.train()
            total_loss = 0.0
            total_pos_pct = 0.0
            optimizer.zero_grad()

            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{args.epochs}"
            )

            for i, (inputs, targets, occluded, _) in progress_bar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.unsqueeze(1).float().to(device, non_blocking=True)
                occluded = occluded.unsqueeze(1).float().to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets, occluded)

                loss = loss / args.accumulation_steps
                loss.backward()

                if ((i + 1) % args.accumulation_steps == 0) or ((i + 1) == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                current_loss = loss.item() * args.accumulation_steps
                total_loss += current_loss

                with torch.no_grad():
                    batch_pos_pct = (torch.sigmoid(outputs) > 0.5).float().mean().item() * 100.0
                    total_pos_pct += batch_pos_pct

                progress_bar.set_postfix(loss=f"{current_loss:.4f}", pos_pct=f"{batch_pos_pct:.1f}%")

            scheduler.step()

            avg_loss = total_loss / len(train_loader)
            avg_pos_pct = round(total_pos_pct / len(train_loader), 2)
            current_lr = scheduler.get_last_lr()[0]
            epoch_duration = time.time() - t_epoch_start

            # 1. Lưu checkpoint định kỳ nhẹ (~131 MB) chỉ chứa trọng số model phục vụ evaluate
            save_path = os.path.join(args.checkpoint_dir, f"swin_amodal_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)

                        # 2. Lưu/ghi đè last.pth đầy đủ (~405 MB: model + optimizer + scheduler) phục vụ resume tiết kiệm ổ đĩa (Ghi nguyên tử)
            last_save_path = os.path.join(args.checkpoint_dir, "last.pth")
            tmp_last_path = last_save_path + ".tmp"
            last_checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            with open(tmp_last_path, "wb") as f_chk:
                torch.save(last_checkpoint_data, f_chk)
                f_chk.flush()
                os.fsync(f_chk.fileno())
            os.replace(tmp_last_path, last_save_path)

            # Đánh giá validation định kỳ
            val_metrics = None
            if val_loader is not None and ((epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs):
                logger.info(f"Đang đánh giá validation tại Epoch {epoch + 1}...")
                val_metrics = evaluate_validation(model, val_loader, device)

            logger.log_epoch(
                epoch=epoch + 1,
                total_epochs=args.epochs,
                train_loss=avg_loss,
                duration_sec=epoch_duration,
                lr=current_lr,
                checkpoint_path=save_path,
                val_metrics=val_metrics,
                train_pos_pct=avg_pos_pct
            )

        # Đánh giá tổng kết sau khi hoàn thành toàn bộ epochs
        target_eval_loader = final_val_loader if final_val_loader is not None else val_loader
        if target_eval_loader is not None and args.eval_final:
            logger.info(f"📊 Đang thực hiện đánh giá cuối cùng trên toàn bộ tập validation ({len(target_eval_loader.dataset)} mẫu)...")
            final_metrics = evaluate_validation(model, target_eval_loader, device)

    except Exception as e:
        logger.log_error(e, epoch=logger.current_epoch)
        raise
    finally:
        logger.log_end(final_metrics=final_metrics)


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Huấn luyện Swin-UNet Row 5 trên D2SA")
    parser.add_argument("--img-dir", type=str, default=os.path.join(root_dir, "data", "D2SA", "images"))
    parser.add_argument(
        "--ann-files",
        nargs="+",
        default=[
            os.path.join(root_dir, "data", "D2SA", "D2S_amodal_training_rot0.json"),
            os.path.join(root_dir, "data", "D2SA", "D2S_amodal_augmented.json"),
        ]
    )
    parser.add_argument(
        "--val-ann-file",
        type=str,
        default=os.path.join(root_dir, "data", "D2SA", "D2S_amodal_validation.json"),
        help="File annotation validation để tính mIoU"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Đường dẫn trực tiếp tới file checkpoint (.pth) cần resume")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--occlusion-weight", type=float, default=5.0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(root_dir, "checkpoints", "d2sa", "amodal_shape_prediction_no_embeding")
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(root_dir, "logs", "d2sa", "row5_occ_edge_noemb_spatial"),
        help="Thư mục lưu log (.log và .jsonl)"
    )
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0, help="Số mẫu val dùng đánh giá (0 = toàn bộ)")
    parser.add_argument("--eval-every", type=int, default=5, help="Chu kỳ epoch để chạy evaluate validation")
    parser.add_argument("--eval-final", action="store_true", default=True, help="Luôn evaluate sau epoch cuối cùng")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
