"""
===================================================================================
RUN ABLATION COCOA - Huấn luyện & Đánh giá các cấu hình Ablation trên COCOA
===================================================================================
Script tự động hóa huấn luyện và đánh giá 7 cấu hình ablation trên COCO-Amodal (91 lớp):
- Row 1: 4ch (RGB+Vis) | No Edge | No Emb | No Spatial | BCE Loss
- Row 2: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | BCE Loss
- Row 3: 5ch (RGB+Vis+Edge) | Edge | No Emb | No Spatial | Occ-Aware Loss (5x)
- Row 4: 5ch (RGB+Vis+Edge) | Edge | Emb (91) | No Spatial | Occ-Aware Loss (5x) [Main]
- Row 5: 5ch (RGB+Vis+Edge) | Edge | No Emb | Spatial Attn | Occ-Aware Loss (5x)
- Row 6: 5ch (RGB+Vis+Edge) | Edge | Emb (91) | Spatial Attn | Occ-Aware Loss (5x) [Full]
- Row 7: 5ch (RGB+Vis+Edge) | Edge | Emb (91) | SGGatedBottleneck | Occ-Aware Loss (5x) [Proposed SG-GBA]

Đặc biệt:
- Cách ly hoàn toàn thư mục lưu trữ checkpoints/logs/results qua tham số --tag hoặc --checkpoint-dir,
  tránh 100% tình trạng ghi đè lên các checkpoint cũ.
- Tự động nạp kiến trúc mô hình tương ứng và điều phối đầu vào (4ch vs 5ch, có vs không có class_ids).
- Tối ưu hóa toàn diện cho GPU A100 (Mixed Precision FP16, TF32, DataLoader num_workers=8).
===================================================================================
"""

import os
import sys
import time
import json
import argparse
import importlib.util
from datetime import datetime

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dataset import AmodalDataset


# ─────────────────────────────────────────────────────────────────────────────────
# 1. ĐỊNH NGHĨA LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────────
class OcclusionAwareLoss(nn.Module):
    """Loss function kết hợp weighted BCE + Dice loss cho vùng bị che khuất."""
    def __init__(self, occlusion_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.occlusion_weight = occlusion_weight

    def forward(self, pred, target, occluded_region):
        weights = 1.0 + (self.occlusion_weight - 1.0) * occluded_region
        weighted_bce = (self.bce(pred, target) * weights).mean()

        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return weighted_bce + dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """Loss tiêu chuẩn BCE + Dice (dùng cho Row 1 & Row 2)."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, occluded_region=None):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        return bce_loss + dice_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────────
# 2. CẤU HÌNH CHI TIẾT 7 ROWS CỦA COCOA
# ─────────────────────────────────────────────────────────────────────────────────
COCOA_CONFIG_MAP = {
    1: {
        "name": "Row 1 (4ch, BCE, No Edge, No Emb, No Spatial)",
        "short_name": "row1_bce_vis_noedge_noemb_nospatial",
        "model_path": os.path.join(SCRIPTS_DIR, "other_config_cocoa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-noedge-bce", "model.py"),
        "channels": 4,
        "use_emb": False,
        "loss_type": "bce",
        "expected_params": 34_352_027,
    },
    2: {
        "name": "Row 2 (5ch, BCE, Edge, No Emb, No Spatial)",
        "short_name": "row2_bce_vis_edge_noemb_nospatial",
        "model_path": os.path.join(SCRIPTS_DIR, "other_config_cocoa", "amodal-shape-prediction-no-spatial-no-embeding-rgb-vis-edge-bce", "model.py"),
        "channels": 5,
        "use_emb": False,
        "loss_type": "bce",
        "expected_params": 34_353_563,
    },
    3: {
        "name": "Row 3 (5ch, OccLoss 5x, Edge, No Emb, No Spatial)",
        "short_name": "row3_occ_edge_noemb_nospatial",
        "model_path": os.path.join(SCRIPTS_DIR, "other_config_cocoa", "amodal-shape-prediction-no-spatial-no-embeding", "model.py"),
        "channels": 5,
        "use_emb": False,
        "loss_type": "occ",
        "expected_params": 34_353_563,
    },
    4: {
        "name": "Row 4 (Main Config: 5ch, OccLoss 5x, Edge, Emb 91, No Spatial)",
        "short_name": "row4_main_config",
        "model_path": os.path.join(SCRIPTS_DIR, "model.py"),
        "channels": 5,
        "use_emb": True,
        "loss_type": "occ",
        "expected_params": 34_423_451,
    },
    5: {
        "name": "Row 5 (5ch, OccLoss 5x, Edge, No Emb, Spatial Attn)",
        "short_name": "row5_occ_edge_noemb_spatial",
        "model_path": os.path.join(SCRIPTS_DIR, "other_config_cocoa", "amodal_shape_prediction_no_embeding", "model.py"),
        "channels": 5,
        "use_emb": False,
        "loss_type": "occ",
        "expected_params": 34_353_661,
    },
    6: {
        "name": "Row 6 (Full Config: 5ch, OccLoss 5x, Edge, Emb 91, Spatial Attn)",
        "short_name": "row6_full_config",
        "model_path": os.path.join(SCRIPTS_DIR, "other_config_cocoa", "amodal_shape_prediction_full_config", "model.py"),
        "channels": 5,
        "use_emb": True,
        "loss_type": "occ",
        "expected_params": 34_423_549,
    },
    7: {
        "name": "Row 7 (Proposed SG-GBA: 5ch, OccLoss 5x, Edge, Emb 91, SGGatedBottleneck)",
        "short_name": "row7_sg_gba_bottleneck",
        "model_path": os.path.join(SCRIPTS_DIR, "other_config_cocoa", "amodal_shape_prediction_sg_gba_bottleneck", "model.py"),
        "channels": 5,
        "use_emb": True,
        "loss_type": "occ",
        "expected_params": 34_867_566,
    },
}


def load_model_for_row(row_idx, device):
    """Nạp lớp AmodalSwinUNet tương ứng với từng cấu hình từ file model.py cục bộ."""
    cfg = COCOA_CONFIG_MAP[row_idx]
    path = cfg["model_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file model tại: {path}")

    spec = importlib.util.spec_from_file_location(f"cocoa_model_row_{row_idx}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Khởi tạo mô hình phù hợp (hỗ trợ num_classes=91 nếu có Category Embedding)
    if cfg["use_emb"]:
        model = mod.AmodalSwinUNet(num_classes=91)
    else:
        model = mod.AmodalSwinUNet()

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   [Model Row {row_idx}] Tổng tham số: {total_params:,} (Kỳ vọng: {cfg['expected_params']:,})")
    return model


# ─────────────────────────────────────────────────────────────────────────────────
# 3. HÀM TÍNH METRICS ĐÁNH GIÁ CHUẨN
# ─────────────────────────────────────────────────────────────────────────────────
def calculate_metrics_single(pred_logits, target, visible, threshold=0.5):
    """Tính các chỉ số IoU, Invisible IoU, Dice, Precision, Recall cho 1 ảnh."""
    p = (torch.sigmoid(pred_logits[0]) > threshold).float()
    t = (target[0] > 0.5).float()
    v = (visible[0] > 0.5).float()

    inter_o = (p * t).sum().item()
    union_o = (p + t).clamp(0, 1).sum().item()
    p_sum = p.sum().item()
    t_sum = t.sum().item()

    iou = inter_o / union_o if union_o > 0 else 1.0
    dice = (2.0 * inter_o) / (p_sum + t_sum) if (p_sum + t_sum) > 0 else 1.0
    precision = inter_o / p_sum if p_sum > 0 else 1.0
    recall = inter_o / t_sum if t_sum > 0 else 1.0

    invisible_target = ((t - v) > 0.5).float()
    pred_invisible = ((p - v) > 0.5).float()
    inv_inter = (pred_invisible * invisible_target).sum().item()
    inv_union = (pred_invisible + invisible_target).clamp(0, 1).sum().item()
    has_occlusion = invisible_target.sum().item() > 0
    inv_iou = inv_inter / inv_union if inv_union > 0 else -1.0

    return {
        "iou": iou,
        "invisible_iou": inv_iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "has_occlusion": has_occlusion,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# 4. HUẤN LUYỆN 1 ROW
# ─────────────────────────────────────────────────────────────────────────────────
def train_single_row(row_idx, args, device, ckpt_dir):
    cfg = COCOA_CONFIG_MAP[row_idx]
    print(f"\n{'='*75}")
    print(f"🔥 BẮT ĐẦU HUẤN LUYỆN: {cfg['name']}")
    print(f"   Thư mục checkpoint: {ckpt_dir}")
    print(f"   Epochs: {args.epochs} | Batch Size: {args.batch_size} | AMP: {args.amp}")
    print(f"{'='*75}")

    os.makedirs(ckpt_dir, exist_ok=True)
    model = load_model_for_row(row_idx, device)

    # Resume nếu có yêu cầu
    start_epoch = 0
    if args.resume_epoch > 0:
        resume_pth = os.path.join(ckpt_dir, f"swin_amodal_epoch_{args.resume_epoch}.pth")
        if os.path.exists(resume_pth):
            ckpt = torch.load(resume_pth, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            start_epoch = args.resume_epoch
            print(f"🔄 Đã nạp checkpoint resume từ: {resume_pth} (Bắt đầu từ Epoch {start_epoch+1})")
        else:
            print(f"⚠️ Không tìm thấy {resume_pth}, huấn luyện từ Epoch 1!")

    # Loss function
    if cfg["loss_type"] == "occ":
        criterion = OcclusionAwareLoss(occlusion_weight=5.0)
    else:
        criterion = BCEDiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # Fast-forward LR scheduler nếu resume
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    # DataLoader
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    train_dataset = AmodalDataset(img_dir=args.train_img_dir, ann_file=args.train_ann_file, transform=train_transform)
    if args.subset_size > 0:
        indices = list(range(min(args.subset_size, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        print(f"🧪 Chế độ subset huấn luyện: {len(train_dataset)} mẫu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Row {row_idx} | Ep {epoch+1}/{args.epochs}")
        for i, (inputs, targets, occluded, class_ids) in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.unsqueeze(1).float().to(device, non_blocking=True)
            occluded = occluded.unsqueeze(1).float().to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)

            # Cắt 4 kênh cho Row 1 (bỏ kênh edge thứ 4)
            if cfg["channels"] == 4:
                inputs_feed = inputs[:, :4, :, :]
            else:
                inputs_feed = inputs

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                if cfg["use_emb"]:
                    outputs = model(inputs_feed, class_ids)
                else:
                    outputs = model(inputs_feed)

                loss = criterion(outputs, targets, occluded)
                loss_scaled = loss / args.accumulation_steps

            scaler.scale(loss_scaled).backward()

            if ((i + 1) % args.accumulation_steps == 0) or ((i + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        cur_lr = scheduler.get_last_lr()[0]
        alpha_info = ""
        if row_idx == 7 and hasattr(model, "sg_bottleneck") and hasattr(model.sg_bottleneck, "alpha"):
            alpha_val = float(model.sg_bottleneck.alpha.item())
            alpha_info = f" | Alpha: {alpha_val:.5f}"

        print(f"✅ Row {row_idx} Ep {epoch+1} hoàn tất | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}{alpha_info}")

        # Lưu checkpoint cuối epoch
        save_path = os.path.join(ckpt_dir, f"swin_amodal_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

    print(f"💾 Hoàn tất huấn luyện Row {row_idx}! Checkpoint lưu tại: {ckpt_dir}\n")


# ─────────────────────────────────────────────────────────────────────────────────
# 5. ĐÁNH GIÁ 1 ROW TRÊN TẬP VALIDATION
# ─────────────────────────────────────────────────────────────────────────────────
def evaluate_single_row(row_idx, args, device, ckpt_dir, results_dir):
    cfg = COCOA_CONFIG_MAP[row_idx]
    ckpt_path = os.path.join(ckpt_dir, f"swin_amodal_epoch_{args.epochs}.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Không tìm thấy {ckpt_path}, thử tìm checkpoint epoch cao nhất...")
        all_ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("swin_amodal_epoch_") and f.endswith(".pth")]
        if not all_ckpts:
            print(f"❌ Không có checkpoint nào trong {ckpt_dir} để đánh giá!")
            return None
        all_ckpts.sort(key=lambda x: int(x.split("_")[-1].replace(".pth", "")))
        ckpt_path = os.path.join(ckpt_dir, all_ckpts[-1])

    print(f"\n🔍 ĐÁNH GIÁ: {cfg['name']}")
    print(f"   Checkpoint: {ckpt_path}")

    os.makedirs(results_dir, exist_ok=True)
    model = load_model_for_row(row_idx, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    val_transform = A.Compose([A.Resize(224, 224)])
    val_dataset = AmodalDataset(img_dir=args.val_img_dir, ann_file=args.val_ann_file, transform=val_transform)
    if args.val_subset_size > 0:
        indices = list(range(min(args.val_subset_size, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)
        print(f"🧪 Chế độ subset đánh giá: {len(val_dataset)} mẫu")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    tot_iou = 0.0
    tot_dice = 0.0
    tot_prec = 0.0
    tot_rec = 0.0
    tot_inv_iou = 0.0
    occ_count = 0
    total_samples = len(val_loader)

    with torch.no_grad():
        for inputs, targets, occluded, class_ids in tqdm(val_loader, desc=f"Eval Row {row_idx}"):
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            visible_mask = inputs[:, 3:4, :, :].to(device)
            class_ids = class_ids.to(device)

            if cfg["channels"] == 4:
                inputs_feed = inputs[:, :4, :, :]
            else:
                inputs_feed = inputs

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                if cfg["use_emb"]:
                    outputs = model(inputs_feed, class_ids)
                else:
                    outputs = model(inputs_feed)

            metrics = calculate_metrics_single(outputs, targets, visible_mask)
            tot_iou += metrics["iou"]
            tot_dice += metrics["dice"]
            tot_prec += metrics["precision"]
            tot_rec += metrics["recall"]
            if metrics["has_occlusion"] and metrics["invisible_iou"] >= 0:
                tot_inv_iou += metrics["invisible_iou"]
                occ_count += 1

    summary = {
        "mIoU": round((tot_iou / max(1, total_samples)) * 100.0, 2),
        "invisible_mIoU": round((tot_inv_iou / max(1, occ_count)) * 100.0, 2) if occ_count > 0 else 0.0,
        "dice": round((tot_dice / max(1, total_samples)) * 100.0, 2),
        "precision": round((tot_prec / max(1, total_samples)) * 100.0, 2),
        "recall": round((tot_rec / max(1, total_samples)) * 100.0, 2),
        "occlusion_samples": occ_count,
        "total_samples": total_samples,
        "checkpoint": ckpt_path
    }

    if row_idx == 7 and hasattr(model, "sg_bottleneck") and hasattr(model.sg_bottleneck, "alpha"):
        summary["gate_alpha"] = round(float(model.sg_bottleneck.alpha.item()), 5)

    eval_result_path = os.path.join(results_dir, f"{cfg['short_name']}_eval.json")
    with open(eval_result_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg["name"], "row": row_idx, "summary": summary}, f, indent=2, ensure_ascii=False)

    print(f"📊 Kết quả Row {row_idx}: mIoU: {summary['mIoU']}% | Inv-mIoU: {summary['invisible_mIoU']}% | Dice: {summary['dice']}%")
    print(f"💾 Đã lưu JSON tại: {eval_result_path}\n")
    return summary


# ─────────────────────────────────────────────────────────────────────────────────
# 6. TỔNG HỢP BÁO CÁO TABLE V (JSON & MARKDOWN)
# ─────────────────────────────────────────────────────────────────────────────────
def generate_table_v_cocoa(results_dir, executed_rows):
    table_records = []
    for r in sorted(executed_rows):
        if r not in COCOA_CONFIG_MAP:
            continue
        cfg = COCOA_CONFIG_MAP[r]
        eval_path = os.path.join(results_dir, f"{cfg['short_name']}_eval.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sm = data.get("summary", {})
                rec = {
                    "row": r,
                    "name": cfg["name"],
                    "mIoU": sm.get("mIoU", 0.0),
                    "inv_mIoU": sm.get("invisible_mIoU", 0.0),
                    "dice": sm.get("dice", 0.0),
                    "precision": sm.get("precision", 0.0),
                    "recall": sm.get("recall", 0.0),
                }
                if "gate_alpha" in sm:
                    rec["gate_alpha"] = sm["gate_alpha"]
                table_records.append(rec)
            except Exception as e:
                print(f"⚠️ Không thể đọc {eval_path}: {e}")

    if not table_records:
        return

    # Lưu JSON tổng hợp
    summary_json = os.path.join(results_dir, "table_v_cocoa_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(table_records, f, indent=2, ensure_ascii=False)

    # Lưu Markdown tổng hợp
    summary_md = os.path.join(results_dir, "table_v_cocoa_summary.md")
    lines = [
        "# 📊 BẢNG TỔNG HỢP SO SÁNH THỰC NGHIỆM ABLATION STUDY COCOA (TABLE V)",
        "",
        "| Row | Cấu hình | Overall mIoU (%) | Invisible mIoU (%) | Dice (%) | Precision (%) | Recall (%) | Gate Alpha |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in table_records:
        alpha_str = str(r.get("gate_alpha", "-"))
        lines.append(
            f"| {r['row']} | {r['name']} | **{r['mIoU']:.2f}** | **{r['inv_mIoU']:.2f}** | {r['dice']:.2f} | {r['precision']:.2f} | {r['recall']:.2f} | {alpha_str} |"
        )
    lines.append("")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 80)
    print("🏆 BÁO CÁO TỔNG HỢP TABLE V COCOA ĐÃ ĐƯỢC XUẤT:")
    print(f"   • JSON     : {summary_json}")
    print(f"   • Markdown : {summary_md}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────────
# 7. MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Master Ablation Runner cho COCOA (7 cấu hình)")
    parser.add_argument("--rows", nargs="+", type=int, default=[4, 6, 7], help="Danh sách Row cần chạy (1–7, mặc định 4 6 7)")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "all"], default="all", help="Chế độ: train, eval, all")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (mặc định 16 cho A100)")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate AdamW")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader num_workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Bật Mixed Precision FP16")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Epoch muốn resume")
    parser.add_argument("--subset-size", type=int, default=0, help="Số mẫu huấn luyện (0 = full)")
    parser.add_argument("--val-subset-size", type=int, default=0, help="Số mẫu validation (0 = full)")

    # Dữ liệu & Lưu trữ
    parser.add_argument("--drive-dir", type=str, default="", help="Thư mục Google Drive (nếu chạy trên Colab)")
    parser.add_argument("--tag", type=str, default="cocoa_ablation_v2", help="Tag thư mục riêng để chống đè")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Đường dẫn custom cho checkpoints")
    parser.add_argument("--results-dir", type=str, default="", help="Đường dẫn custom cho results")

    # Đường dẫn dữ liệu
    parser.add_argument("--train-img-dir", type=str, default=os.path.join(ROOT_DIR, "data", "train2014"))
    parser.add_argument("--train-ann-file", type=str, default=os.path.join(ROOT_DIR, "data", "annotations", "COCO_amodal_train2014.json"))
    parser.add_argument("--val-img-dir", type=str, default=os.path.join(ROOT_DIR, "data", "val2014"))
    parser.add_argument("--val-ann-file", type=str, default=os.path.join(ROOT_DIR, "data", "annotations", "COCO_amodal_val2014.json"))

    args = parser.parse_args()

    # Thiết bị
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Thiết lập thư mục lưu trữ độc lập
    base_dir = os.path.abspath(args.drive_dir) if args.drive_dir else ROOT_DIR
    base_ckpt = args.checkpoint_dir if args.checkpoint_dir else os.path.join(base_dir, "checkpoints", args.tag)
    base_res = args.results_dir if args.results_dir else os.path.join(base_dir, "results", args.tag)

    os.makedirs(base_ckpt, exist_ok=True)
    os.makedirs(base_res, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 MASTER ABLATION RUNNER CHO COCOA (7 ROWS)")
    print(f"   • Rows thực thi     : {args.rows}")
    print(f"   • Chế độ            : {args.mode}")
    print(f"   • Thiết bị          : {device} (AMP: {args.amp})")
    print(f"   • Checkpoints Folder: {base_ckpt}")
    print(f"   • Results Folder    : {base_res}")
    print(f"   • Train Ann File    : {args.train_ann_file}")
    print(f"   • Val Ann File      : {args.val_ann_file}")
    print("=" * 80)

    success_rows = []
    for r in sorted(args.rows):
        if r not in COCOA_CONFIG_MAP:
            print(f"⚠️ Row {r} không hợp lệ, bỏ qua.")
            continue

        cfg = COCOA_CONFIG_MAP[r]
        row_ckpt_dir = os.path.join(base_ckpt, cfg["short_name"])

        # Train
        if args.mode in ["train", "all"]:
            try:
                train_single_row(r, args, device, row_ckpt_dir)
            except Exception as e:
                print(f"❌ Lỗi khi huấn luyện Row {r}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Eval
        if args.mode in ["eval", "all"]:
            try:
                summary = evaluate_single_row(r, args, device, row_ckpt_dir, base_res)
                if summary:
                    success_rows.append(r)
            except Exception as e:
                print(f"❌ Lỗi khi đánh giá Row {r}: {e}")
                import traceback
                traceback.print_exc()

    # Tổng hợp Table V
    if args.mode in ["eval", "all"] and success_rows:
        generate_table_v_cocoa(base_res, success_rows)


if __name__ == "__main__":
    main()

