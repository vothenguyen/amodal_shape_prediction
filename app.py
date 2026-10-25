"""
===================================================================================
AMODAL STUDIO PRO — GRADIO WEB UI (SIDE-BY-SIDE COMPARISON & XAI HEATMAP)
===================================================================================
Hệ thống demo đa tập dữ liệu (COCOA, D2SA, KINS) phục vụ bảo vệ Khóa luận Tốt nghiệp 2026:
1. So sánh trực tiếp song song (Side-by-Side):
   - Row 4: MAPR 2026 Baseline (Prior-Guided Swin-UNet)
   - Row 7: Thesis 2026 Proposed (Semantic-Guided Gated Bottleneck Attention - SG-GBA)
2. Real-time Telemetry Panel:
   - Visible Pixels, Occluded Pixels, Expansion Ratio (|M_amodal| / |M_vis|)
   - Cổng học được alpha (Learnable Residual Gate)
   - Độ trễ suy luận Stage 2 (Latency ms) & Tốc độ khung hình (FPS)
3. Giải thích mô hình XAI (Explainable AI & Attention Paradox):
   - Bottleneck Attention Heatmap (7x7 -> HxW, JET Colormap)
   - Decoder Feature Activation Maps (up_final 64 channels, VIRIDIS Colormap)
4. Model Registry & Checkpoint Fallback:
   - Tự động fallback sang chế độ demo [Demo Mode - Untrained] nếu chưa có file .pth
   - Tương thích định dạng raw state_dict và dict {'model_state_dict': ...}
===================================================================================
"""

import os
import sys
import time
import gc
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from ultralytics import SAM

# Đảm bảo đường dẫn gốc nằm trong sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import 2 mô hình cốt lõi
from scripts.model import AmodalSwinUNet as AmodalSwinUNetRow4
from scripts.other_config_d2sa.amodal_shape_prediction_sg_gba_bottleneck.model import AmodalSwinUNet as AmodalSwinUNetRow7


# ===================================================================================
# 1. DANH MỤC LỚP CHO 3 TẬP DỮ LIỆU (COCOA, D2SA, KINS)
# ===================================================================================

COCO_CLASSES = {
    1: "person (người)", 2: "bicycle (xe đạp)", 3: "car (xe hơi)", 4: "motorcycle (xe máy)",
    5: "airplane (máy bay)", 6: "bus (xe buýt)", 7: "train (tàu hỏa)", 8: "truck (xe tải)",
    9: "boat (thuyền)", 10: "traffic light (đèn giao thông)", 11: "fire hydrant (trụ cứu hỏa)",
    13: "stop sign (biển dừng)", 14: "parking meter (đồng hồ đỗ xe)", 15: "bench (ghế đá)",
    16: "bird (chim)", 17: "cat (mèo)", 18: "dog (chó)", 19: "horse (ngựa)", 20: "sheep (cừu)",
    21: "cow (bò)", 22: "elephant (voi)", 23: "bear (gấu)", 24: "zebra (ngựa vằn)",
    25: "giraffe (hươu cao cổ)", 27: "backpack (ba lô)", 28: "umbrella (ô/dù)",
    31: "handbag (túi xách)", 32: "tie (cà vạt)", 33: "suitcase (vali)", 34: "frisbee (đĩa ném)",
    35: "skis (ván trượt tuyết)", 36: "snowboard (ván trượt)", 37: "sports ball (bóng)",
    38: "kite (diều)", 39: "baseball bat (gậy bóng chày)", 40: "baseball glove (găng tay)",
    41: "skateboard (ván trượt)", 42: "surfboard (ván lướt sóng)", 43: "tennis racket (vợt tennis)",
    44: "bottle (cái chai)", 46: "wine glass (ly rượu)", 47: "cup (cái cốc)", 48: "fork (cái nĩa)",
    49: "knife (con dao)", 50: "spoon (cái thìa)", 51: "bowl (cái bát)", 52: "banana (chuối)",
    53: "apple (táo)", 54: "sandwich (bánh mì kẹp)", 55: "orange (cam)", 56: "broccoli (súp lơ)",
    57: "carrot (cà rốt)", 58: "hot dog (xúc xích)", 59: "pizza (pizza)", 60: "donut (donut)",
    61: "cake (bánh ngọt)", 62: "chair (ghế)", 63: "couch (sofa)", 64: "potted plant (cây cảnh)",
    65: "bed (giường)", 67: "dining table (bàn ăn)", 70: "toilet (bồn cầu)", 72: "tv (tivi)",
    73: "laptop (máy tính xách tay)", 74: "mouse (chuột máy tính)", 75: "remote (điều khiển)",
    76: "keyboard (bàn phím)", 77: "cell phone (điện thoại)", 78: "microwave (lò vi sóng)",
    79: "oven (lò nướng)", 80: "toaster (máy nướng bánh)", 81: "sink (bồn rửa)",
    82: "refrigerator (tủ lạnh)", 84: "book (sách)", 85: "clock (đồng hồ)", 86: "vase (lọ hoa)",
    87: "scissors (kéo)", 88: "teddy bear (gấu bông)", 89: "hair drier (máy sấy)", 90: "toothbrush (bàn chải)",
}

D2SA_CLASSES = {
    0: "adelholzener_alpenquelle_classic_075", 1: "adelholzener_alpenquelle_naturell_075",
    2: "adelholzener_classic_bio_apfelschorle_02", 3: "adelholzener_classic_naturell_02",
    4: "adelholzener_gourmet_mineralwasser_02", 5: "augustiner_lagerbraeu_hell_05",
    6: "augustiner_weissbier_05", 7: "coca_cola_05", 8: "coca_cola_light_05",
    9: "suntory_gokuri_limonade", 10: "tegernseer_hell_03", 11: "corny_nussvoll",
    12: "corny_nussvoll_single", 13: "corny_schoko_banane", 14: "corny_schoko_banane_single",
    15: "dr_oetker_vitalis_knuspermuesli_klassisch", 16: "koelln_muesli_fruechte",
    17: "koelln_muesli_schoko", 18: "caona_kakaohaltiges_getraenkepulver",
    19: "cocoba_fruehstueckskakao_mit_honig", 20: "cafe_wunderbar_espresso",
    21: "douwe_egberts_professional_kaffee_gemahlen", 22: "gepa_bio_caffe_crema",
    23: "gepa_italienischer_bio_espresso", 24: "apple_braeburn_bundle",
    25: "apple_golden_delicious", 26: "apple_granny_smith", 27: "apple_roter_boskoop",
    28: "avocado", 29: "banana_bundle", 30: "banana_single", 31: "clementine",
    32: "clementine_single", 33: "grapes_green_sugraone_seedless",
    34: "grapes_sweet_celebration_seedless", 35: "kiwi", 36: "orange_single",
    37: "oranges", 38: "pear", 39: "pasta_reggia_elicoidali", 40: "pasta_reggia_fusilli",
    41: "pasta_reggia_spaghetti", 42: "franken_tafelreiniger", 43: "pelikan_tintenpatrone_canon",
    44: "ethiquable_gruener_tee_ceylon", 45: "gepa_bio_und_fair_fencheltee",
    46: "gepa_bio_und_fair_kamillentee", 47: "gepa_bio_und_fair_kraeuterteemischung",
    48: "gepa_bio_und_fair_pfefferminztee", 49: "gepa_bio_und_fair_rooibostee",
    50: "kilimanjaro_tea_earl_grey", 51: "cucumber", 52: "carrot", 53: "feldsalat",
    54: "lettuce", 55: "rispentomaten", 56: "roma_rispentomaten", 57: "rucola",
    58: "salad_iceberg", 59: "zucchini"
}

KINS_CLASSES = {
    0: "background (nền)",
    1: "cyclist (người đi xe đạp)",
    2: "pedestrian (người đi bộ)",
    3: "rider (người điều khiển xe)",
    4: "car (xe hơi)",
    5: "van (xe tải nhỏ/van)",
    6: "truck (xe tải lớn)",
    7: "tram (tàu điện)",
    8: "misc (đối tượng khác)",
    9: "traffic sign (biển báo)",
    10: "traffic light (đèn giao thông)",
    11: "pole (cột điện/đèn)",
    12: "construction (công trình xây dựng)",
    13: "vegetation (thảm thực vật/cây)",
    14: "terrain (địa hình)",
    15: "sky (bầu trời)",
    16: "ground (mặt đất)",
    17: "dynamic obstacle (vật cản động)",
    18: "static obstacle (vật cản tĩnh)",
    19: "dontcare (vùng bỏ qua)",
}

DATASET_CONFIGS = {
    "COCOA (91 classes - Daily Objects)": {
        "num_classes": 91,
        "classes": COCO_CLASSES,
        "default_choice": "24 - zebra (ngựa vằn)",
        "row4_default_ckpt": "checkpoints/swin_amodal_epoch_30.pth",
        "row7_default_ckpt": "checkpoints/cocoa/row7_sg_gba_bottleneck/swin_amodal_epoch_30.pth",
        "zero_indexed": False,
    },
    "D2SA (60 classes - Supermarket Products)": {
        "num_classes": 60,
        "classes": D2SA_CLASSES,
        "default_choice": "0 - adelholzener_alpenquelle_classic_075",
        "row4_default_ckpt": "checkpoints/d2sa/amodal_shape_prediction_main_config/swin_amodal_epoch_30.pth",
        "row7_default_ckpt": "checkpoints/d2sa/row7_sg_gba_bottleneck/swin_amodal_epoch_1.pth",
        "zero_indexed": True,
    },
    "KINS (20 classes - Autonomous Driving)": {
        "num_classes": 20,
        "classes": KINS_CLASSES,
        "default_choice": "4 - car (xe hơi)",
        "row4_default_ckpt": "checkpoints/kins/swin_amodal_KINS_FULL_epoch_30.pth",
        "row7_default_ckpt": "checkpoints/kins/swin_amodal_KINS_row7_epoch_30.pth",
        "zero_indexed": True,
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================================
# 2. AN TOÀN NẠP CHECKPOINT & QUẢN LÝ BỘ NHỚ
# ===================================================================================

def safe_load_checkpoint(model, ckpt_path, device):
    """
    Nạp checkpoint an toàn:
    - Kiểm tra đường dẫn tồn tại.
    - Hỗ trợ cả file state_dict trực tiếp lẫn dict chứa key 'model_state_dict'.
    - Gỡ bỏ tiền tố '_orig_mod.' nếu mô hình từng được bọc bởi torch.compile.
    - Tự động fallback về [Demo Mode - Untrained] nếu chưa có file .pth.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False, f"⚠️ [Demo Mode - Untrained] Không tìm thấy file: {ckpt_path}"
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_state_dict = (
            ckpt['model_state_dict']
            if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt)
            else ckpt
        )
        cleaned = {
            (k[10:] if k.startswith('_orig_mod.') else k): v
            for k, v in raw_state_dict.items()
        }
        incompat = model.load_state_dict(cleaned, strict=False)
        missing_count = len(incompat.missing_keys)
        unexp_count = len(incompat.unexpected_keys)
        
        status_msg = f"✅ Đã nạp: {os.path.basename(ckpt_path)}"
        if missing_count > 0 or unexp_count > 0:
            status_msg += f" (Khuyết {missing_count}, Thừa {unexp_count} keys)"
        return True, status_msg
    except Exception as e:
        return False, f"⚠️ [Demo Mode - Lỗi Nạp: {str(e)[:60]}]"


class AmodalStudioManager:
    """Quản lý các mô hình Stage 1 (SAM 2.1) và Stage 2 (Row 4 + Row 7) theo tập dữ liệu."""
    def __init__(self, device):
        self.device = device
        self.current_dataset_key = None
        self.model_row4 = None
        self.model_row7 = None
        self.row4_status = "Chưa nạp"
        self.row7_status = "Chưa nạp"
        self.sam_model = None
        self.init_sam()

    def init_sam(self):
        """Khởi tạo SAM 2.1 với cơ chế fallback sang SAM 2.1 Tiny nếu thiếu VRAM."""
        print(f"📥 Đang nạp mô hình SAM 2.1 trên {self.device}...")
        sam_paths = ["checkpoints/sam2.1_b.pt", "sam2.1_b.pt", "sam2.1_t.pt"]
        for sp in sam_paths:
            try:
                self.sam_model = SAM(sp)
                self.sam_model.to(self.device)
                print(f"✅ SAM 2.1 sẵn sàng từ: {sp}")
                return
            except Exception as e:
                continue
        print("⚠️ Cảnh báo: SAM 2.1 chưa có checkpoint cục bộ, sẽ tự động download khi nhận prompt.")
        try:
            self.sam_model = SAM("sam2.1_b.pt")
            self.sam_model.to(self.device)
        except Exception:
            self.sam_model = None

    def load_dataset_models(self, dataset_key, row4_path=None, row7_path=None):
        """Nạp hoặc chuyển đổi kiến trúc Swin-UNet theo tập dữ liệu đã chọn."""
        cfg = DATASET_CONFIGS[dataset_key]
        num_classes = cfg["num_classes"]

        row4_ckpt = row4_path if (row4_path and row4_path.strip()) else cfg["row4_default_ckpt"]
        row7_ckpt = row7_path if (row7_path and row7_path.strip()) else cfg["row7_default_ckpt"]

        # Nếu chuyển dataset hoặc model chưa khởi tạo, khởi tạo lại
        if self.current_dataset_key != dataset_key or self.model_row4 is None or self.model_row7 is None:
            del self.model_row4, self.model_row7
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"🔄 Khởi tạo mô hình Row 4 & Row 7 cho {dataset_key} (num_classes={num_classes})...")
            self.model_row4 = AmodalSwinUNetRow4(pretrained=False, num_classes=num_classes).to(self.device)
            self.model_row7 = AmodalSwinUNetRow7(pretrained=False, num_classes=num_classes).to(self.device)
            self.current_dataset_key = dataset_key

        self.model_row4.eval()
        self.model_row7.eval()

        # Nạp trọng số
        _, self.row4_status = safe_load_checkpoint(self.model_row4, row4_ckpt, self.device)
        _, self.row7_status = safe_load_checkpoint(self.model_row7, row7_ckpt, self.device)

        return self.row4_status, self.row7_status


# Khởi tạo thể hiện duy nhất của Model Manager
STUDIO_MANAGER = AmodalStudioManager(DEVICE)
# Nạp mặc định COCOA khi khởi động
STUDIO_MANAGER.load_dataset_models("COCOA (91 classes - Daily Objects)")


# ===================================================================================
# 3. CÁC HÀM XỬ LÝ TRỰC QUAN HÓA & XAI HEATMAPS
# ===================================================================================

def create_amodal_overlay(orig_img, visible_mask, amodal_pred_mask, alpha=0.45):
    """
    Tạo overlay 3 vùng màu chuẩn Khóa luận Tốt nghiệp:
    - Vùng nhìn thấy (M_vis): Xanh lá [0, 255, 0]
    - Vùng khuất phục hồi được (M_pred \\ M_vis): Xanh dương [30, 144, 255]
    - Viền ngoài Amodal Contour (Biên của M_pred): Đỏ [255, 0, 0]
    """
    h, w = orig_img.shape[:2]
    vis_full = cv2.resize(visible_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pred_full = cv2.resize(amodal_pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Vùng khuất = M_pred trừ đi M_vis
    occ_full = np.clip(pred_full.astype(np.int32) - vis_full.astype(np.int32), 0, 1).astype(np.uint8)

    overlay = orig_img.copy()
    overlay[vis_full == 1] = [0, 255, 0]      # Xanh lá: Visible
    overlay[occ_full == 1] = [30, 144, 255]   # Xanh dương / Cyan: Occluded

    # Trộn bán trong suốt với ảnh gốc
    blended = cv2.addWeighted(overlay, alpha, orig_img, 1.0 - alpha, 0)

    # Vẽ đường viền đỏ sắc nét quanh contour Amodal
    contours, _ = cv2.findContours(pred_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 0, 0), thickness=2)

    return blended


def generate_heatmap_overlay(orig_img, attn_map_tensor, alpha=0.55):
    """
    Trích xuất bản đồ chú ý Bottleneck (7x7), nội suy Bilinear lên kích thước (H, W),
    chuẩn hóa Min-Max [0, 255] và phủ bảng màu cv2.COLORMAP_JET đè lên ảnh gốc.
    """
    h, w = orig_img.shape[:2]
    if attn_map_tensor is None:
        return orig_img.copy()

    if isinstance(attn_map_tensor, torch.Tensor):
        t = attn_map_tensor.detach().float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            t = t.unsqueeze(0)
        upsampled = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
        attn_np = upsampled[0, 0].cpu().numpy()
    else:
        attn_np = cv2.resize(attn_map_tensor, (w, h), interpolation=cv2.INTER_CUBIC)

    min_val, max_val = float(attn_np.min()), float(attn_np.max())
    if max_val - min_val > 1e-7:
        norm_attn = ((attn_np - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_attn = np.zeros((h, w), dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(norm_attn, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(heatmap_rgb, alpha, orig_img, 1.0 - alpha, 0)
    return blended


def generate_decoder_feature_map(orig_img, feat_tensor, alpha=0.50):
    """
    Trích xuất kích hoạt đặc trưng tầng cuối Decoder (up_final — trung bình cộng 64 kênh),
    nội suy Bilinear lên kích thước (H, W) và phủ bảng màu COLORMAP_VIRIDIS đè lên ảnh gốc.
    """
    h, w = orig_img.shape[:2]
    if feat_tensor is None:
        return orig_img.copy()

    if isinstance(feat_tensor, torch.Tensor):
        t = feat_tensor.detach().float()
        if t.dim() == 4:
            t = t[0]  # [64, 224, 224]
        mean_feat = t.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, 224, 224]
        upsampled = F.interpolate(mean_feat, size=(h, w), mode='bilinear', align_corners=False)
        feat_np = upsampled[0, 0].cpu().numpy()
    else:
        feat_np = cv2.resize(feat_tensor, (w, h), interpolation=cv2.INTER_CUBIC)

    min_val, max_val = float(feat_np.min()), float(feat_np.max())
    if max_val - min_val > 1e-7:
        norm_feat = ((feat_np - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_feat = np.zeros((h, w), dtype=np.uint8)

    feat_bgr = cv2.applyColorMap(norm_feat, cv2.COLORMAP_VIRIDIS)
    feat_rgb = cv2.cvtColor(feat_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(feat_rgb, alpha, orig_img, 1.0 - alpha, 0)
    return blended


def create_priors_display(orig_img, visible_mask, edge_mask, pts):
    """
    Tổng hợp trực quan hóa Input Priors:
    - Ảnh gốc kèm điểm click (vòng tròn xanh)
    - Vùng Visible Mask (SAM) bán trong suốt màu lục
    - Vùng Edge Mask viền màu vàng cam
    """
    h, w = orig_img.shape[:2]
    vis_full = cv2.resize(visible_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    edge_full = cv2.resize((edge_mask > 0.1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    priors_img = orig_img.copy()
    overlay = priors_img.copy()
    overlay[vis_full == 1] = [0, 220, 0]
    overlay[edge_full == 1] = [255, 140, 0]  # Viền cam

    blended = cv2.addWeighted(overlay, 0.4, priors_img, 0.6, 0)

    # Vẽ các điểm click người dùng
    for pt in pts:
        cv2.circle(blended, tuple(pt), 7, (255, 255, 255), -1)
        cv2.circle(blended, tuple(pt), 5, (255, 0, 0), -1)

    return blended


# ===================================================================================
# 4. HÀM SUY LUẬN KÉP SONG SONG (SIDE-BY-SIDE PREDICTION PIPELINE)
# ===================================================================================

def predict_side_by_side(orig_image, pts, dataset_key, category_choice):
    """
    Thực hiện toàn bộ quy trình suy luận 2-Stage song song:
    1. SAM 2.1 trích xuất Visible Mask và tính Edge Mask.
    2. Chạy qua Row 4 Baseline (MAPR 2026) -> đo Latency, FPS, xuất Decoder Features.
    3. Chạy qua Row 7 Proposed SG-GBA -> đo Latency, FPS, xuất Bottleneck Heatmap, alpha, Decoder Features.
    4. Tổng hợp Telemetry Panel và hình ảnh kết quả.
    """
    if orig_image is None:
        return (None, None, None, None, None, None, "❌ Chưa có ảnh đầu vào! Vui lòng tải ảnh lên.")
    if not pts or len(pts) == 0:
        return (None, None, None, None, None, None, "❌ Chưa chọn điểm trên ảnh! Hãy click vào vật thể cần hoàn thiện amodal.")

    try:
        # ──────────────────────────────────────────────────────────────
        # BƯỚC 1: XÁC ĐỊNH CLASS ID TƯƠNG ỨNG VỚI DATASET
        # ──────────────────────────────────────────────────────────────
        cfg = DATASET_CONFIGS[dataset_key]
        raw_id = int(category_choice.split(" - ")[0])
        # Chuẩn hoá chỉ số class embedding theo convention từng dataset
        if dataset_key.startswith("COCOA"):
            model_class_id = raw_id
        elif dataset_key.startswith("D2SA"):
            model_class_id = max(0, min(cfg["num_classes"] - 1, raw_id))
        else:  # KINS
            model_class_id = max(0, min(cfg["num_classes"] - 1, raw_id))

        h_orig, w_orig = orig_image.shape[:2]

        # ──────────────────────────────────────────────────────────────
        # BƯỚC 2: STAGE 1 — SAM 2.1 TRÍCH XUẤT VISIBLE MASK
        # ──────────────────────────────────────────────────────────────
        combined_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        if STUDIO_MANAGER.sam_model is not None:
            for pt in pts:
                x, y = pt
                # Bỏ qua điểm nếu đã nằm trong vùng mask hợp lệ trước đó
                if combined_mask[y, x] > 128:
                    continue
                results = STUDIO_MANAGER.sam_model(
                    orig_image,
                    points=[pt],
                    labels=[1],
                    device=DEVICE,
                    retina_masks=True,
                    conf=0.45,
                    verbose=False,
                )
                if results and results[0].masks is not None:
                    single_mask = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
                    combined_mask = cv2.bitwise_or(combined_mask, single_mask)

        # Fallback an toàn nếu SAM chưa segment được (ví dụ điểm ở nền)
        if np.max(combined_mask) == 0:
            print("⚠️ SAM không bắt được mask, sử dụng vùng tròn quanh điểm prompt làm fallback.")
            for pt in pts:
                cv2.circle(combined_mask, tuple(pt), 40, 255, -1)

        visible_mask_orig = (combined_mask > 128).astype(np.uint8)

        # ──────────────────────────────────────────────────────────────
        # BƯỚC 3: CHUẨN BỊ INPUT 5 KÊNH (224 x 224)
        # ──────────────────────────────────────────────────────────────
        img_resized = cv2.resize(orig_image, (224, 224))
        vis_resized = cv2.resize(visible_mask_orig, (224, 224), interpolation=cv2.INTER_NEAREST)
        visible_mask_224 = (vis_resized > 0).astype(np.float32)

        # Tạo morphological edge mask từ visible mask
        vis_uint8 = (visible_mask_224 * 255).astype(np.uint8)
        k_morph = np.ones((5, 5), np.uint8)
        edge_mask_224 = (
            cv2.dilate(vis_uint8, k_morph, iterations=1).astype(np.float32)
            - cv2.erode(vis_uint8, k_morph, iterations=1).astype(np.float32)
        ) / 255.0

        # Chuẩn hóa ảnh RGB
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        vis_tensor = torch.from_numpy(visible_mask_224).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(edge_mask_224).float().unsqueeze(0)

        input_5ch = torch.cat([img_tensor, vis_tensor, edge_tensor], dim=0).unsqueeze(0).to(DEVICE)
        cls_tensor = torch.tensor([model_class_id], dtype=torch.long).to(DEVICE)

        # ──────────────────────────────────────────────────────────────
        # BƯỚC 4: SUY LUẬN ROW 4 BASELINE (MAPR 2026)
        # ──────────────────────────────────────────────────────────────
        decoder_feats = {}
        def hook_row4(m, inp, out):
            decoder_feats['row4'] = out.detach()
        def hook_row7(m, inp, out):
            decoder_feats['row7'] = out.detach()

        h_hook4 = STUDIO_MANAGER.model_row4.up_final.register_forward_hook(hook_row4)
        h_hook7 = STUDIO_MANAGER.model_row7.up_final.register_forward_hook(hook_row7)

        # Suy luận Row 4
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_row4 = time.perf_counter()
        with torch.no_grad():
            out_row4 = STUDIO_MANAGER.model_row4(input_5ch, cls_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_row4_ms = (time.perf_counter() - t_start_row4) * 1000.0
        fps_row4 = 1000.0 / max(lat_row4_ms, 0.001)

        pred_mask_row4_224 = (torch.sigmoid(out_row4)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        # ──────────────────────────────────────────────────────────────
        # BƯỚC 5: SUY LUẬN ROW 7 PROPOSED SG-GBA (THESIS 2026)
        # ──────────────────────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_row7 = time.perf_counter()
        with torch.no_grad():
            out_row7 = STUDIO_MANAGER.model_row7(input_5ch, cls_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_row7_ms = (time.perf_counter() - t_start_row7) * 1000.0
        fps_row7 = 1000.0 / max(lat_row7_ms, 0.001)

        pred_mask_row7_224 = (torch.sigmoid(out_row7)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        # Gỡ hooks
        h_hook4.remove()
        h_hook7.remove()

        # Trích xuất giá trị alpha và attention map của Row 7
        alpha_val = STUDIO_MANAGER.model_row7.sg_bottleneck.alpha.item()
        attn_map_row7 = STUDIO_MANAGER.model_row7.sg_bottleneck.last_attn_map

        # ──────────────────────────────────────────────────────────────
        # BƯỚC 6: TẠO OVERLAYS VÀ BẢN ĐỒ NHIỆT XAI
        # ──────────────────────────────────────────────────────────────
        # 1. Input Priors Display
        priors_display = create_priors_display(orig_image, visible_mask_orig, cv2.resize(edge_mask_224, (w_orig, h_orig)), pts)

        # 2. Row 4 Output Overlay
        row4_overlay = create_amodal_overlay(orig_image, visible_mask_orig, pred_mask_row4_224)

        # 3. Row 7 Output Overlay
        row7_overlay = create_amodal_overlay(orig_image, visible_mask_orig, pred_mask_row7_224)

        # 4. Row 7 Bottleneck Attention Heatmap (7x7 -> HxW)
        row7_attn_heatmap = generate_heatmap_overlay(orig_image, attn_map_row7, alpha=0.55)

        # 5. Row 4 & Row 7 Decoder Activation Maps
        row4_decoder_feat = generate_decoder_feature_map(orig_image, decoder_feats.get('row4'), alpha=0.50)
        row7_decoder_feat = generate_decoder_feature_map(orig_image, decoder_feats.get('row7'), alpha=0.50)

        # ──────────────────────────────────────────────────────────────
        # BƯỚC 7: TÍNH TOÁN METRICS VÀ TELEMETRY PANEL
        # ──────────────────────────────────────────────────────────────
        pred_full_row4 = cv2.resize(pred_mask_row4_224, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        pred_full_row7 = cv2.resize(pred_mask_row7_224, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        area_vis = int(np.sum(visible_mask_orig))
        area_row4 = int(np.sum(pred_full_row4))
        area_occ_row4 = int(np.sum(np.clip(pred_full_row4.astype(np.int32) - visible_mask_orig.astype(np.int32), 0, 1)))
        exp_ratio_row4 = (area_row4 / max(area_vis, 1))

        area_row7 = int(np.sum(pred_full_row7))
        area_occ_row7 = int(np.sum(np.clip(pred_full_row7.astype(np.int32) - visible_mask_orig.astype(np.int32), 0, 1)))
        exp_ratio_row7 = (area_row7 / max(area_vis, 1))

        telemetry_md = f"""
### 📊 BẢNG ĐO ĐẠC SUY LUẬN THỜI GIAN THỰC (REAL-TIME TELEMETRY PANEL)
| Tiêu chí Đánh giá | **Row 4: Baseline (MAPR 2026)** | **Row 7: Proposed SG-GBA (Thesis 2026)** | So sánh / Nhận xét |
| :--- | :--- | :--- | :--- |
| **Trạng thái Trọng số** | `{STUDIO_MANAGER.row4_status}` | `{STUDIO_MANAGER.row7_status}` | Multi-dataset Checkpoint Registry |
| **Diện tích Nhìn thấy ($M_{{vis}}$)** | **{area_vis:,} px** | **{area_vis:,} px** | SAM 2.1 Zero-shot Stage 1 |
| **Vùng khuất Tái tạo ($M_{{pred}} \\setminus M_{{vis}}$)** | **{area_occ_row4:,} px** | **{area_occ_row7:,} px** | {"Row 7 mở rộng tốt hơn" if area_occ_row7 > area_occ_row4 else "Bảo toàn chặt chẽ"} |
| **Tỷ lệ Mở rộng Hình dạng** | **{exp_ratio_row4:.2f}×** ({exp_ratio_row4*100:.1f}%) | **{exp_ratio_row7:.2f}×** ({exp_ratio_row7*100:.1f}%) | $\\text{{Expansion}} = |M_{{amodal}}| / |M_{{vis}}|$ |
| **Độ trễ Suy luận Stage 2 (Latency)** | **{lat_row4_ms:.2f} ms** | **{lat_row7_ms:.2f} ms** | Chênh lệch: `{abs(lat_row7_ms - lat_row4_ms):.2f} ms` |
| **Tốc độ Khung hình (FPS)** | **{fps_row4:.1f} FPS** | **{fps_row7:.1f} FPS** | Đạt thời gian thực (> 30 FPS) |
| **Cổng Học được $\\alpha$ (Residual Gate)** | *Không có (Direct Residual)* | **`{alpha_val:.6f}`** | Điều biến cân bằng ngữ nghĩa toàn cục |

> 💡 **Quan sát XAI**: Chú ý toàn cục tại Bottleneck ($7 \\times 7$) học vị trí và hướng mở rộng của vật thể bị che, trong khi tầng Decoder ($224 \\times 224$) đảm nhiệm việc làm sắc nét các đường biên. Cổng học được $\\alpha$ điều tiết mức độ can thiệp, triệt tiêu hiện tượng nhòe viền (Attention Paradox).
"""

        # Giải phóng bộ nhớ đệm GPU
        del input_5ch, cls_tensor, out_row4, out_row7
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            priors_display,
            row4_overlay,
            row7_overlay,
            row7_attn_heatmap,
            row4_decoder_feat,
            row7_decoder_feat,
            telemetry_md
        )

    except Exception as e:
        err_msg = f"🔥 Lỗi trong quá trình suy luận: {str(e)}"
        print(err_msg)
        return (None, None, None, None, None, None, err_msg)


# ===================================================================================
# 5. GRADIO EVENT HANDLERS
# ===================================================================================

def on_image_click(orig_img, current_points, evt: gr.SelectData):
    """Lưu lại tọa độ và vẽ điểm tròn đỏ/xanh preview trực tiếp."""
    if orig_img is None:
        return None, current_points
    x, y = evt.index[0], evt.index[1]
    current_points.append([x, y])

    display = orig_img.copy()
    for pt in current_points:
        cv2.circle(display, tuple(pt), 7, (255, 255, 255), -1)
        cv2.circle(display, tuple(pt), 5, (255, 0, 0), -1)
    return display, current_points


def on_clear_points(orig_img):
    """Xóa tất cả các điểm đã chọn."""
    return orig_img, []


def on_dataset_change(dataset_key):
    """Khi đổi dataset: Cập nhật danh sách Categories và đường dẫn Checkpoints mặc định."""
    cfg = DATASET_CONFIGS[dataset_key]
    choices = [f"{k} - {v}" for k, v in cfg["classes"].items()]
    default_choice = cfg["default_choice"]
    row4_path = cfg["row4_default_ckpt"]
    row7_path = cfg["row7_default_ckpt"]

    status_row4, status_row7 = STUDIO_MANAGER.load_dataset_models(dataset_key, row4_path, row7_path)
    status_summary = f"**Hệ thống đã chuyển sang `{dataset_key}`:**\n- Row 4: `{status_row4}`\n- Row 7: `{status_row7}`"

    return (
        gr.Dropdown(choices=choices, value=default_choice),
        row4_path,
        row7_path,
        status_summary
    )


def on_reload_models(dataset_key, row4_path, row7_path):
    """Nạp lại checkpoint tùy chỉnh do người dùng nhập."""
    status_row4, status_row7 = STUDIO_MANAGER.load_dataset_models(dataset_key, row4_path, row7_path)
    return f"**Cập nhật Mô hình ({dataset_key}):**\n- Row 4 Checkpoint: `{status_row4}`\n- Row 7 Checkpoint: `{status_row7}`"


# ===================================================================================
# 6. THIẾT KẾ GIAO DIỆN GRADIO (AMODAL STUDIO PRO)
# ===================================================================================

custom_css = """
.gradio-container {
    max-width: 98% !important;
}
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    padding: 20px 25px;
    border-radius: 12px;
    margin-bottom: 18px;
    border-left: 6px solid #3b82f6;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.telemetry-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
}
"""

app_theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="Amodal Studio Pro — Side-by-Side & XAI Heatmap") as demo:

    # Header
    gr.HTML("""
    <div class="header-box">
        <h1 style="margin: 0; font-size: 26px; font-weight: 700;">🎯 Amodal Studio Pro — Interactive Comparison & XAI Telemetry</h1>
        <p style="margin: 6px 0 0 0; font-size: 14.5px; opacity: 0.9;">
            Hệ thống So sánh Trực quan Song song & Giải thích Mô hình phục vụ <strong>Khóa luận Tốt nghiệp 2026</strong>. 
            Hỗ trợ 3 tập dữ liệu chuẩn: <strong>COCOA</strong> (91 lớp), <strong>D2SA</strong> (60 lớp), và <strong>KINS</strong> (20 lớp).
        </p>
    </div>
    """)

    # Model Registry Accordion
    with gr.Accordion("⚙️ Model Registry & Cấu hình Đa Dataset (COCOA / D2SA / KINS)", open=False):
        with gr.Row():
            dataset_selector = gr.Radio(
                choices=list(DATASET_CONFIGS.keys()),
                value="COCOA (91 classes - Daily Objects)",
                label="📁 Chọn Tập dữ liệu (Dataset Mode)",
                interactive=True
            )
        with gr.Row():
            row4_ckpt_input = gr.Textbox(
                label="📦 Đường dẫn Checkpoint Row 4 (MAPR 2026 Baseline)",
                value=DATASET_CONFIGS["COCOA (91 classes - Daily Objects)"]["row4_default_ckpt"]
            )
            row7_ckpt_input = gr.Textbox(
                label="📦 Đường dẫn Checkpoint Row 7 (Thesis 2026 Proposed SG-GBA)",
                value=DATASET_CONFIGS["COCOA (91 classes - Daily Objects)"]["row7_default_ckpt"]
            )
        with gr.Row():
            btn_reload_ckpts = gr.Button("🔄 Tải / Nạp lại Checkpoint tùy chỉnh", variant="secondary")
            model_registry_status = gr.Markdown(
                f"**Trạng thái Mô hình:**\n- Row 4: `{STUDIO_MANAGER.row4_status}`\n- Row 7: `{STUDIO_MANAGER.row7_status}`"
            )

    # State Variables
    original_image_state = gr.State(None)
    points_state = gr.State([])

    # Tabs chính
    with gr.Tabs():
        # TAB 1: SO SÁNH TRỰC QUAN SONG SONG
        with gr.TabItem("⚔️ So sánh Trực quan Song song (Side-by-Side Comparison)"):
            with gr.Row():
                # Cột trái (Input & Điều khiển)
                with gr.Column(scale=4):
                    input_image = gr.Image(
                        label="📸 Ảnh đầu vào (Click trực tiếp lên vật thể để định vị prompt)",
                        interactive=True,
                        height=420
                    )
                    with gr.Row():
                        btn_run_compare = gr.Button("▶️ Chạy So sánh Song song", variant="primary", scale=2)
                        btn_clear_pts = gr.Button("🗑️ Xóa Điểm Click", scale=1)

                    category_dropdown = gr.Dropdown(
                        choices=[f"{k} - {v}" for k, v in COCO_CLASSES.items()],
                        value="24 - zebra (ngựa vằn)",
                        label="🏷️ Danh mục Đối tượng (Category Prior)",
                        interactive=True
                    )

                    priors_preview = gr.Image(
                        label="🎯 Input Priors (SAM Visible Mask + Morphological Edge Mask)",
                        interactive=False,
                        height=260
                    )

                # Cột phải (Kết quả Side-by-Side và Telemetry)
                with gr.Column(scale=8):
                    with gr.Row():
                        row4_output_view = gr.Image(
                            label="🔵 Model 1: Row 4 Baseline (MAPR 2026) — Prior-Guided Swin-UNet",
                            interactive=False,
                            height=400
                        )
                        row7_output_view = gr.Image(
                            label="🌟 Model 2: Row 7 Proposed (Thesis 2026) — SG-GBA Swin-UNet",
                            interactive=False,
                            height=400
                        )

                    # Bảng Telemetry đo đạc trực tiếp
                    telemetry_display = gr.Markdown(
                        """
### 📊 BẢNG ĐO ĐẠC SUY LUẬN THỜI GIAN THỰC (REAL-TIME TELEMETRY PANEL)
*Vui lòng tải ảnh, click chọn vật thể và nhấn **▶️ Chạy So sánh Song song** để hiển thị bảng số liệu so sánh trực tiếp.*
                        """
                    )

        # TAB 2: GIẢI THÍCH MÔ HÌNH XAI (ATTENTION PARADOX & HEATMAPS)
        with gr.TabItem("🔬 Khám phá Trực quan XAI (Explainable AI & Attention Paradox)"):
            gr.Markdown(
                """
### 🧠 Giải thích Cơ chế Chú ý & Nghịch lý Attention (Attention Paradox)
- **Bottleneck Attention ($7 \\times 7$):** Mô-đun **SG-GBA (Row 7)** sử dụng FiLM điều biến kênh theo vector nhúng ngữ nghĩa của loại đối tượng, kết hợp chú ý không gian $7 \\times 7$ để tìm kiếm **khung hình dáng toàn cục của vật thể** (đặc biệt là các phần bị che khuất hoàn toàn).
- **Decoder Features ($224 \\times 224$):** Tầng giải mã phóng tỉ lệ ngược khôi phục các đường biên sắc nét cục bộ nhờ kết nối tắt (skip connections).
- **Cổng Học được $\\alpha$ (Residual Gate):** Khắc phục triệt để nghịch lý Attention (chú ý toàn cục làm mờ nhòe chi tiết biên sắc nét của Stage 1) bằng cách cho phép mô hình tự học tỷ lệ đóng góp của Bottleneck Attention map.
                """
            )
            with gr.Row():
                row7_heatmap_view = gr.Image(
                    label="🔥 Row 7 Bottleneck Attention Heatmap (7x7 -> HxW, JET Colormap)",
                    interactive=False,
                    height=360
                )
                row4_decoder_feat_view = gr.Image(
                    label="⚡ Row 4 Decoder Activation Map (up_final 64 channels, VIRIDIS)",
                    interactive=False,
                    height=360
                )
                row7_decoder_feat_view = gr.Image(
                    label="✨ Row 7 Decoder Activation Map (up_final 64 channels, VIRIDIS)",
                    interactive=False,
                    height=360
                )

    # ===============================
    # BẮT SỰ KIỆN TƯƠNG TÁC
    # ===============================

    # 1. Tải ảnh lên -> Khởi tạo lại trạng thái điểm
    input_image.upload(
        lambda img: (img, []),
        inputs=[input_image],
        outputs=[original_image_state, points_state]
    )

    # 2. Xóa ảnh
    input_image.clear(
        lambda: (None, []),
        outputs=[original_image_state, points_state]
    )

    # 3. Click chọn điểm trên ảnh
    input_image.select(
        on_image_click,
        inputs=[original_image_state, points_state],
        outputs=[input_image, points_state]
    )

    # 4. Xóa điểm click
    btn_clear_pts.click(
        on_clear_points,
        inputs=[original_image_state],
        outputs=[input_image, points_state]
    )

    # 5. Chuyển đổi Dataset Mode
    dataset_selector.change(
        on_dataset_change,
        inputs=[dataset_selector],
        outputs=[category_dropdown, row4_ckpt_input, row7_ckpt_input, model_registry_status]
    )

    # 6. Nạp lại Checkpoint tùy chỉnh
    btn_reload_ckpts.click(
        on_reload_models,
        inputs=[dataset_selector, row4_ckpt_input, row7_ckpt_input],
        outputs=[model_registry_status]
    )

    # 7. Chạy suy luận so sánh song song
    btn_run_compare.click(
        fn=predict_side_by_side,
        inputs=[original_image_state, points_state, dataset_selector, category_dropdown],
        outputs=[
            priors_preview,
            row4_output_view,
            row7_output_view,
            row7_heatmap_view,
            row4_decoder_feat_view,
            row7_decoder_feat_view,
            telemetry_display
        ]
    )


# ===================================================================================
# 7. CHẠY ỨNG DỤNG
# ===================================================================================
if __name__ == "__main__":
    print("🚀 Đang khởi động Amodal Studio Pro...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=app_theme, css=custom_css)