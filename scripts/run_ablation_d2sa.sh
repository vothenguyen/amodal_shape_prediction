#!/usr/bin/env bash
# ===================================================================================
# BASH RUNNER CHO ABLATION STUDY D2SA (CHẠY TRÊN NVIDIA H200 141GB / LINUX / SLURM)
# ===================================================================================
set -e

# Tự động chuyển về thư mục gốc của project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Cấu hình tham số (có thể override bằng biến môi trường hoặc tham số dòng lệnh)
DEVICE="${1:-${DEVICE:-cuda}}"
EPOCHS="${2:-${EPOCHS:-30}}"
BATCH_SIZE="${3:-${BATCH_SIZE:-16}}"
ACC_STEPS="${4:-${ACC_STEPS:-1}}"
NUM_WORKERS="${5:-${NUM_WORKERS:-8}}"
VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-500}"
ROWS="${ROWS:-1 2 3 4 5 6}"

echo "========================================================================"
echo "🚀 ABLATION STUDY D2SA (MVTec D2S Amodal) - H200 RUNNER (141GB VRAM)"
echo "   Thư mục dự án:  ${ROOT_DIR}"
echo "   Các Row chạy:   ${ROWS}"
echo "   Thiết bị GPU:   ${DEVICE}"
echo "   Số Epoch:       ${EPOCHS}"
echo "   Batch Size:     ${BATCH_SIZE} (Tích lũy ${ACC_STEPS} bước -> Effective Batch = $((BATCH_SIZE * ACC_STEPS)))"
echo "   DataLoader:     ${NUM_WORKERS} workers"
echo "   Val Subset Định kỳ: ${VAL_SUBSET_SIZE} mẫu (Epoch 30 cuối cùng sẽ evaluate full 15,654 mẫu)"
echo "========================================================================"

# Chạy runner python chính
python scripts/run_ablation_d2sa.py \
    --rows ${ROWS} \
    --mode all \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulation-steps "${ACC_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --val-subset-size "${VAL_SUBSET_SIZE}" \
    --device "${DEVICE}"

echo ""
echo "========================================================================"
echo "✅ Hoàn tất toàn bộ các cấu hình ablation trên D2SA!"
echo "   Xem log chi tiết tại: logs/d2sa/"
echo "   Xem checkpoint tại:   checkpoints/d2sa/"
echo "========================================================================"
