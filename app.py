"""
===================================================================================
GIAO DIỆN GRADIO - AMODAL SHAPE PREDICTION DEMO
===================================================================================
Ứng dụng web interactivedemonstrate mô hình dự đoán hình dạng amodal.

Tính năng:
- Tải ảnh lên
- Click chọn vật thể trên ảnh (hỗ trợ click nhiều lần)
- Chọn loại vật thể từ dropdown (91 loại COCO)
- Pipeline 2-stage:
  1. SAM 2.1: Phát hiện vật thể từ click
  2. Swin-UNet: Dự đoán hình dạng amodal
- Hiển thị:
  1. Kết quả từ SAM (phần nhìn thấy)
  2. Kết quả từ UNet (dự đoán hình dạng toàn bộ)
  3. Thống kê và trạng thái

Chạy: python app.py
===================================================================================
"""

import gradio as gr
from gradio.themes import Base
import torch
import cv2
import numpy as np
from ultralytics import SAM
import gc

from src.model import AmodalSwinUNet


# ===================================================================================
# DANH SÁCH 90 LOẠI VẬT THỂ COCO
# ===================================================================================
COCO_CLASSES = {
    1: "person (người)",
    2: "bicycle (xe đạp)",
    3: "car (xe hơi)",
    4: "motorcycle (xe máy)",
    5: "airplane (máy bay)",
    6: "bus (xe buýt)",
    7: "train (tàu hỏa)",
    8: "truck (xe tải)",
    9: "boat (thuyền)",
    10: "traffic light (đèn giao thông)",
    11: "fire hydrant (trụ cứu hỏa)",
    13: "stop sign (biển dừng)",
    14: "parking meter (đồng hồ đỗ xe)",
    15: "bench (ghế đá)",
    16: "bird (chim)",
    17: "cat (mèo)",
    18: "dog (chó)",
    19: "horse (ngựa)",
    20: "sheep (cừu)",
    21: "cow (bò)",
    22: "elephant (voi)",
    23: "bear (gấu)",
    24: "zebra (ngựa vằn)",
    25: "giraffe (hươu cao cổ)",
    27: "backpack (ba lô)",
    28: "umbrella (ô/dù)",
    31: "handbag (túi xách)",
    32: "tie (cà vạt)",
    33: "suitcase (vali)",
    34: "frisbee (đĩa ném)",
    35: "skis (ván trượt tuyết)",
    36: "snowboard (ván trượt tuyết)",
    37: "sports ball (bóng thể thao)",
    38: "kite (diều)",
    39: "baseball bat (gậy bóng chày)",
    40: "baseball glove (găng tay)",
    41: "skateboard (ván trượt)",
    42: "surfboard (ván lướt sóng)",
    43: "tennis racket (vợt tennis)",
    44: "bottle (cái chai)",
    46: "wine glass (ly rượu)",
    47: "cup (cái cốc)",
    48: "fork (cái nĩa)",
    49: "knife (con dao)",
    50: "spoon (cái thìa)",
    51: "bowl (cái bát)",
    52: "banana (quả chuối)",
    53: "apple (quả táo)",
    54: "sandwich (bánh sandwich)",
    55: "orange (quả cam)",
    56: "broccoli (súp lơ)",
    57: "carrot (cà rốt)",
    58: "hot dog (xúc xích)",
    59: "pizza (bánh pizza)",
    60: "donut (bánh donut)",
    61: "cake (bánh ngọt)",
    62: "chair (cái ghế)",
    63: "couch (ghế sofa)",
    64: "potted plant (cây chậu)",
    65: "bed (cái giường)",
    67: "dining table (bàn ăn)",
    70: "toilet (bồn cầu)",
    72: "tv (tivi)",
    73: "laptop (máy tính xách tay)",
    74: "mouse (chuột máy tính)",
    75: "remote (điều khiển)",
    76: "keyboard (bàn phím)",
    77: "cell phone (điện thoại)",
    78: "microwave (lò vi sóng)",
    79: "oven (lò nướng)",
    80: "toaster (máy nướng bánh)",
    81: "sink (bồn rửa)",
    82: "refrigerator (tủ lạnh)",
    84: "book (quyển sách)",
    85: "clock (đồng hồ)",
    86: "vase (lọ hoa)",
    87: "scissors (cái kéo)",
    88: "teddy bear (gấu bông)",
    89: "hair drier (máy sấy tóc)",
    90: "toothbrush (bàn chải)",
}

# Tạo danh sách dropdown: "ID - Tên"
CLASS_CHOICES = [f"{k} - {v}" for k, v in COCO_CLASSES.items()]

# Chọn thiết bị: GPU nếu có, không thì CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================================
# NẠP CÁC MÔ HÌNH
# ===================================================================================
print("📥 Đang nạp SAM 2.1...")
sam_model = SAM("checkpoints/sam2.1_b.pt")
sam_model.to(DEVICE)  # Chuyển SAM sang GPU

print("📥 Đang nạp Swin-UNet...")
amodal_model = AmodalSwinUNet(num_classes=91).to(DEVICE)
amodal_model.load_state_dict(
    torch.load("checkpoints/swin_amodal_epoch_30.pth", map_location=DEVICE)
)
amodal_model.eval()  # Bật chế độ evaluation (không training)


# ===================================================================================
# XỨLÝ SỰ KIỆN NGƯỜI DÙNG
# ===================================================================================
def get_point(orig_img, current_points, evt: gr.SelectData):
    """
    Lưu điểm khi người dùng click trên ảnh.
    
    Args:
        orig_img: Ảnh gốc đã tải lên
        current_points: Danh sách điểm đã click
        evt: Sự kiện click từ Gradio (chứa tọa độ)
    
    Returns:
        Tuple: (ảnh cập nhật với điểm vừa click, danh sách điểm cập nhật)
    """
    if orig_img is None:
        return None, current_points

    # Lấy tọa độ (x, y) từ sự kiện click
    x, y = evt.index[0], evt.index[1]
    current_points.append([x, y])

    # Vẽ tất cả điểm lên ảnh để hiển thị preview
    display_img = orig_img.copy()
    for pt in current_points:
        # Vẽ hình tròn xanh dương tại mỗi điểm
        cv2.circle(display_img, tuple(pt), 20, (255, 0, 0), -1)

    return display_img, current_points


def clear_points(orig_img):
    """
    Xóa toàn bộ điểm đã chọn.
    
    Args:
        orig_img: Ảnh gốc
    
    Returns:
        Tuple: (ảnh gốc không có điểm, danh sách điểm rỗng)
    """
    return orig_img, []


# ===================================================================================
# PIPELINE CHÍNH - 2-STAGE (SAM + Swin-UNet)
# ===================================================================================
def end_to_end_predict(orig_image, pts, category_id):
    """
    Pipeline hoàn chỉnh: SAM → Swin-UNet → Hiển thị kết quả.
    
    Quy trình:
    1. PHASE 1 (SAM): Phát hiện vật thể từ click → mask nhị phân
    2. PHASE 2 (Swin-UNet): Dự đoán hình dạng amodal từ visible mask
    3. PHASE 3: Hiển thị kết quả và thống kê
    
    Args:
        orig_image: Ảnh gốc từ ứng dụng
        pts: Danh sách điểm click [[x1, y1], [x2, y2], ...]
        category_id: ID loại vật thể từ dropdown
    
    Returns:
        Tuple:
        - sam_colored: Ảnh hiển thị SAM mask
        - result_img: Ảnh hiển thị dự đoán amodal
        - debug_status: Chuỗi thông tin thống kê
    """
    try:
        # ──────────────────────────────────────────────────────────────
        # KIỂM TRA INPUT
        # ──────────────────────────────────────────────────────────────
        if orig_image is None or not pts:
            return None, None, "❌ Chưa có ảnh hoặc chưa chọn điểm!"

        # Rút trích ID từ chuỗi "24 - zebra (ngựa vằn)"
        real_cat_id = int(category_id.split(" - ")[0])
        # SAM yêu cầu labels (1 = foreground, 0 = background)
        labels = [1] * len(pts)

        # ──────────────────────────────────────────────────────────────
        # PHASE 1: SAM - PHÁT HIỆN VẬT THỂ (XỬ LÝ ĐA VÙNG BỊ CHE KHUẤT)
        # ──────────────────────────────────────────────────────────────
        print(f"🎯 SAM: Bắt đầu xử lý {len(pts)} điểm click...")
        
        # Khởi tạo một mask tổng (chứa toàn bộ nền đen)
        h, w = orig_image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Duyệt qua từng điểm người dùng đã click
        for i, pt in enumerate(pts):
            x, y = pt
            
            # TỐI ƯU HÓA THEO LOGIC CỦA SẾP: 
            # Nếu điểm click này đã nằm bên trong vùng mask vừa được segment trước đó,
            # ta sẽ bỏ qua không gọi SAM nữa để tiết kiệm hàng tỷ phép tính của GPU.
            if combined_mask[y, x] > 128:
                print(f"   ↳ Bỏ qua điểm {i+1} {pt} vì đã nằm trong vùng cắt hợp lệ.")
                continue

            print(f"   ↳ Đang cắt riêng vùng cho điểm {i+1}: {pt}...")
            # Gọi SAM chỉ với 1 điểm duy nhất
            results = sam_model(
                orig_image,
                points=[pt],       # Truyền 1 điểm
                labels=[1],        # Foreground label
                device=DEVICE,
                retina_masks=True, # Lấy mask chất lượng cao
                conf=0.45,
                verbose=False,
            )

            # Nếu cắt thành công, gộp mask mới vào mask tổng
            if results and results[0].masks is not None:
                single_mask = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
                
                # Phép Hợp Logic (Logical Union - Bitwise OR): Gộp vùng mới vào vùng cũ
                combined_mask = cv2.bitwise_or(combined_mask, single_mask)

        # Kiểm tra xem sau khi gộp, mask tổng có chứa dữ liệu không
        if np.max(combined_mask) == 0:
            return None, None, "❌ SAM không tìm thấy vật thể từ các điểm chỉ định."

        # Cập nhật mask tổng để đưa xuống Phase 2 (Swin-UNet)
        sam_mask_full = combined_mask
        print("✅ SAM: Hoàn tất trích xuất và gộp mask!")

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: SWIN-UNET - DỰĐOÁN HÌNH DẠNG AMODAL
        # ──────────────────────────────────────────────────────────────
        print("🧠 Swin-UNet: Dự đoán hình dạng amodal...")
        
        # Resize ảnh về 224×224 (kích thước mô hình)
        img_resized = cv2.resize(orig_image, (224, 224))
        mask_resized = cv2.resize(sam_mask_full, (224, 224))

        # Chuyển mask binary về float [0, 1]
        visible_mask = (mask_resized > 128).astype(np.float32)

        # ─────────────────────────────────────────────────
        # Tạo EDGE MASK (viền gợi ý)
        # ─────────────────────────────────────────────────
        visible_uint8 = (visible_mask * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        # Edge = dilate - erode (viền ranh giới)
        edge_mask = (
            cv2.dilate(visible_uint8, kernel, iterations=1)
            - cv2.erode(visible_uint8, kernel, iterations=1)
        ) / 255.0

        # ─────────────────────────────────────────────────
        # Chuẩn hóa ảnh RGB
        # ─────────────────────────────────────────────────
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Tạo 5-kênh input
        vis_tensor = torch.from_numpy(visible_mask).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(edge_mask).float().unsqueeze(0)

        input_tensor = torch.cat(
            [img_tensor, vis_tensor, edge_tensor], dim=0
        ).unsqueeze(0).to(DEVICE)  # [1, 5, 224, 224]

        # Chuyển class ID thành tensor
        class_id_tensor = torch.tensor([real_cat_id]).long().to(DEVICE)

        # Dự đoán
        with torch.no_grad():
            output = amodal_model(input_tensor, class_id_tensor)
            pred_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # ──────────────────────────────────────────────────────────────
        # PHASE 3: HIỂN THỊ KẾT QUẢ
        # ──────────────────────────────────────────────────────────────
        
        # 1. Hiển thị SAM mask (Overlay xanh lá bán trong suốt)
        sam_colored = img_resized.copy()
        sam_overlay = sam_colored.copy()
        
        sam_overlay[visible_mask == 1] = [0, 255, 0]  # Tô xanh lá trên layer ảo
        # Trộn layer ảo với ảnh gốc (Độ trong suốt 40%)
        cv2.addWeighted(sam_overlay, 0.4, sam_colored, 0.6, 0, sam_colored)

        # 2. Hiển thị kết quả Amodal (Overlay xanh lá + vàng)
        result_img = img_resized.copy()
        amodal_overlay = result_img.copy()

        # Xác định vùng bị che khuất
        invisible_part = np.clip(pred_mask - visible_mask, 0, 1)

        # Tô màu trên CÙNG MỘT layer ảo
        amodal_overlay[visible_mask == 1] = [0, 255, 0]       # Phần nhìn thấy = Xanh lá (giống hệt SAM)
        amodal_overlay[invisible_part == 1] = [255, 255, 0]   # Phần bị lấp = Vàng

        # Trộn layer ảo (chứa cả xanh và vàng) với ảnh gốc
        cv2.addWeighted(amodal_overlay, 0.4, result_img, 0.6, 0, result_img)

        # ──────────────────────────────────────────────────────────────
        # TÍNH THỐNG KÊ
        # ──────────────────────────────────────────────────────────────
        area_sam = int(np.sum(visible_mask))
        area_swin = int(np.sum(pred_mask))
        area_invisible = int(np.sum(invisible_part))

        # Xác định xem có occlusion không
        THRESHOLD = 200  # Ngưỡng số pixel để xác định occlusion
        status_msg = (
            "Có phần bị che khuất."
            if area_invisible > THRESHOLD
            else "Không có phần bị che khuất."
        )

        debug_status = (
            f"📊 SAM: {area_sam} px | Swin: {area_swin} px | Hidden: {area_invisible} px. {status_msg}"
        )

        # ──────────────────────────────────────────────────────────────
        # GIẢI PHÓNG BỘ NHỚ GPU
        # ──────────────────────────────────────────────────────────────
        del input_tensor, class_id_tensor, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sam_colored, result_img, debug_status

    except Exception as e:
        return None, None, f"🔥 Lỗi: {str(e)}"


# ===================================================================================
# GIAO DIỆN GRADIO
# ===================================================================================
with gr.Blocks() as demo:
    # Tiêu đề
    gr.Markdown("# 👁️ Amodal Shape Predictor - Dự đoán hình dạng vật thể bị che khuất")

    # Quản lý state (trạng thái)
    original_image = gr.State(None)      # Ảnh gốc
    point_state = gr.State([])          # Danh sách điểm click

    with gr.Row():
        # ─────── CỘT TRÁI: INPUT ───────
        with gr.Column():
            # Tải ảnh lên
            input_image = gr.Image(label="📸 Ảnh đầu vào", interactive=True)

            # Nút điều khiển
            with gr.Row():
                btn_clear = gr.Button("🗑️ Xóa điểm")
                btn_run = gr.Button("▶️ Chạy dự đoán")

            # Chọn loại vật thể
            cat_dropdown = gr.Dropdown(
                choices=CLASS_CHOICES,
                value="24 - zebra (ngựa vằn)",
                label="🏷️ Chọn loại vật thể",
                filterable=True,
            )

        # ─────── CỘT PHẢI: OUTPUT ───────
        with gr.Column():
            sam_output = gr.Image(label="🎯 Kết quả SAM (Vùng phát hiện)")
            final_output = gr.Image(label="🧠 Kết quả Amodal (Dự đoán hình dạng)")
            status = gr.Textbox(label="📊 Trạng thái")

    # ─────────────────────────────────────────────────────────────
    # SỰ KIỆN NGƯỜI DÙNG
    # ─────────────────────────────────────────────────────────────
    # Khi tải ảnh lên
    input_image.upload(
        lambda img: (img, []),
        inputs=[input_image],
        outputs=[original_image, point_state],
    )

    # Khi xóa ảnh
    input_image.clear(lambda: (None, []), outputs=[original_image, point_state])

    # Khi click trên ảnh
    input_image.select(
        get_point,
        inputs=[original_image, point_state],
        outputs=[input_image, point_state],
    )

    # Khi nhấn nút Xóa điểm
    btn_clear.click(
        clear_points,
        inputs=[original_image],
        outputs=[input_image, point_state],
    )

    # Khi nhấn nút Chạy dự đoán
    btn_run.click(
        fn=end_to_end_predict,
        inputs=[original_image, point_state, cat_dropdown],
        outputs=[sam_output, final_output, status],
    )


if __name__ == "__main__":
    demo.launch(theme=Base(), share=True)