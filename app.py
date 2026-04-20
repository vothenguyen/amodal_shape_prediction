import gradio as gr
import torch
import cv2
import numpy as np
from ultralytics import SAM
import gc

from src.model import AmodalSwinUNet

# TỪ ĐIỂN COCO CHUẨN ĐẦY ĐỦ 80 LỚP
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
CLASS_CHOICES = [f"{k} - {v}" for k, v in COCO_CLASSES.items()]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. NẠP MÔ HÌNH (CHUẨN XỊN NHẤT)
# ==========================================
print("⏳ Đang nạp Mắt Thần SAM (Bản Base - Thông minh, bám viền gắt)...")
sam_model = SAM("sam2_b.pt")
sam_model.to("cpu")

print("⏳ Đang nạp Não Tưởng Tượng Swin-UNet (Epoch 30)...")
amodal_model = AmodalSwinUNet(num_classes=91).to(DEVICE)
amodal_model.load_state_dict(
    torch.load("checkpoints/swin_amodal_epoch_30.pth", map_location=DEVICE)
)
amodal_model.eval()


# ==========================================
# 2. HÀM XỬ LÝ (CHỐNG LỖI CHẤM ĐỎ)
# ==========================================
def get_point(orig_img, current_points, evt: gr.SelectData):
    if orig_img is None:
        return None, current_points

    x, y = evt.index[0], evt.index[1]
    current_points.append([x, y])

    # Chỉ vẽ chấm đỏ lên MÀN HÌNH cho user xem, giữ nguyên ảnh gốc
    display_img = orig_img.copy()
    for pt in current_points:
        cv2.circle(display_img, tuple(pt), radius=6, color=(255, 0, 0), thickness=-1)

    return display_img, current_points


def clear_points(orig_img):
    return orig_img, []


def end_to_end_predict(orig_image, pts, category_id):
    try:
        if orig_image is None or not pts:
            return (
                None,
                None,
                "❌ Cảnh báo: Cậu chưa tải ảnh hoặc chưa CLICK CHUỘT vào vật thể!",
            )

        real_cat_id = int(category_id.split(" - ")[0])

        # Gán nhãn Foreground (1) cho tất cả các điểm cậu đã click
        labels = [1] * len(pts)

        # Đưa ảnh GỐC SẠCH vào SAM. Conf=0.45 để nó lọc nhiễu cực mạnh
        results = sam_model(
            orig_image,
            points=pts,
            labels=labels,
            device="cpu",
            retina_masks=True,
            conf=0.45,
            verbose=False,
        )

        if not results or results[0].masks is None:
            return (
                None,
                None,
                "❌ Lỗi: SAM không tìm thấy vật thể. Hãy bấm 'Xóa điểm' và chọn lại!",
            )

        sam_mask_full = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)

        # --- PHASE 2: SWIN-UNET TƯỞNG TƯỢNG (Trên GPU) ---
        img_resized = cv2.resize(orig_image, (224, 224))
        mask_resized = cv2.resize(sam_mask_full, (224, 224))
        
        visible_mask = (mask_resized > 128).astype(np.float32)

        visible_uint8 = (visible_mask * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(visible_uint8, kernel, iterations=1)
        erosion = cv2.erode(visible_uint8, kernel, iterations=1)
        edge_mask = (dilation - erosion) / 255.0

        # CẢNH BÁO BẮT MẠCH: Nếu lúc train cậu xài Normalize ImageNet thì mở comment 3 dòng dưới ra nhé!
        img_norm = img_resized.astype(np.float32) / 255.0
        # mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        # std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        # img_tensor = (torch.from_numpy(img_norm.transpose(2, 0, 1)) - torch.tensor(mean)) / torch.tensor(std)
        
        # Nếu train không xài Normalize thì xài dòng mặc định này:
        # --- ÉP CHUẨN IMAGENET ĐỂ CHỮA BỆNH CHO SWIN-UNET ---
        # img_norm = img_resized.astype(np.float32) / 255.0
        # mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        # std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        # Trừ mean, chia std đúng bài bản của timm
        # img_tensor = (torch.from_numpy(img_norm.transpose(2, 0, 1)) - torch.tensor(mean)) / torch.tensor(std)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.float()
        
        vis_tensor = torch.from_numpy(visible_mask).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(edge_mask).float().unsqueeze(0)
        
        input_tensor = torch.cat([img_tensor, vis_tensor, edge_tensor], dim=0).unsqueeze(0).to(DEVICE)
        class_id_tensor = torch.tensor([real_cat_id], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = amodal_model(input_tensor, class_id_tensor)
            pred_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # --- PHASE 3: RENDER X-RAY (Chụp X-Quang Swin-UNet) ---
        sam_colored = img_resized.copy()
        sam_colored[visible_mask == 1] = [0, 255, 0]  # RGB: Xanh lá
        
        result_img = img_resized.copy()
        
        # 1. Phủ màu Đỏ mờ cho TOÀN BỘ hình dáng Swin-UNet tưởng tượng ra
        red_overlay = result_img.copy()
        red_overlay[pred_mask == 1] = [255, 0, 0]  # RGB: Đỏ
        cv2.addWeighted(red_overlay, 0.4, result_img, 0.6, 0, result_img)

        # 2. Tô màu Vàng rực cho riêng phần bị khuất (Invisible)
        invisible_part = np.clip(pred_mask - visible_mask, 0, 1)
        result_img[invisible_part == 1] = [255, 255, 0]  # RGB: Vàng

        # 3. Ép nó khai ra đang có bao nhiêu Pixel!
        area_sam = int(np.sum(visible_mask))
        area_swin = int(np.sum(pred_mask))
        area_invisible = int(np.sum(invisible_part))
        if area_invisible > 0:
            status_msg = "Swin-UNet đã dự đoán được phần che khuất."
        else:
            status_msg = "Swin-UNet chưa phát hiện thêm phần che khuất; có thể vật thể quá nhỏ hoặc model cần tinh chỉnh."
        debug_status = (
            f"✅ Phân tích: SAM cắt được {area_sam} px | Swin tưởng tượng ra {area_swin} px | "
            f"Khuất: {area_invisible} px. {status_msg}"
        )

        del input_tensor, class_id_tensor, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sam_colored, result_img, debug_status
    
    except Exception as e:
        return None, None, f"🔥 LỖI CODE: {str(e)}"


# ==========================================
# 3. GIAO DIỆN WEB
# ==========================================
with gr.Blocks() as demo:
    # TIÊU ĐỀ MỚI - NHÌN VÀO LÀ BIẾT CHẠY ĐÚNG FILE
    gr.Markdown("# 👁️ End-to-End Amodal Shape Predictor (BẢN V4.0 - HOÀN HẢO)")
    gr.Markdown(
        "**Cập nhật:** Đã sửa lỗi nhiễu chấm đỏ. Giờ đây cậu có thể click nhiều điểm để lấy trọn con ngựa vằn!"
    )

    # 2 biến State bí mật giữ ảnh sạch và danh sách tọa độ
    original_image = gr.State(None)
    point_state = gr.State([])

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Ảnh Gốc (Click nhiều điểm tùy thích)", interactive=True
            )

            with gr.Row():
                btn_clear = gr.Button("🔄 Xóa điểm chọn", variant="secondary")
                btn_run = gr.Button("🚀 Khai Hỏa (SAM 2 Base)", variant="primary")

            cat_dropdown = gr.Dropdown(
                choices=CLASS_CHOICES,
                value="24 - zebra (ngựa vằn)",
                label="🔍 Tìm kiếm ID Vật Thể (Gõ tên để tìm nhanh)",
                interactive=True,
                filterable=True,
            )

        with gr.Column():
            sam_output = gr.Image(label="Kênh 4 (SAM Tự Cắt - Xanh lá)")
            final_output = gr.Image(label="Swin-UNet Phục Chế (Bị khuất - Vàng)")
            status = gr.Textbox(label="Bảng Trạng Thái")

    # Khi up ảnh mới, lưu ngay 1 bản SẠCH vào original_image
    input_image.upload(
        lambda img: (img, []),
        inputs=[input_image],
        outputs=[original_image, point_state],
    )
    input_image.clear(lambda: (None, []), outputs=[original_image, point_state])

    # Khi click, chỉ vẽ lên màn hình, không đụng tới original_image
    input_image.select(
        get_point,
        inputs=[original_image, point_state],
        outputs=[input_image, point_state],
    )

    btn_clear.click(
        clear_points, inputs=[original_image], outputs=[input_image, point_state]
    )

    # Khi chạy, đưa bản SẠCH vào SAM
    btn_run.click(
        fn=end_to_end_predict,
        inputs=[original_image, point_state, cat_dropdown],
        outputs=[sam_output, final_output, status],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), share=True)
