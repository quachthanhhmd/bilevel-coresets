#!/bin/bash

# ==============================================================================
# Script chạy thực nghiệm cho bài toán Continual Learning (Học liên tục)
# Có hỗ trợ truyền tham số từ dòng lệnh (Command Line Arguments).
# ==============================================================================

# Giá trị mặc định
DATASET="splitmnist"
METHODS=("uniform" "kmeans_features" "sensitivity" "glister" "coreset")
SEEDS=(0 1 2)
BUFFER_SIZES=(50 100 200)
BETAS=(0.01 1.0 100.0)
STAGE="all"

# Hàm in hướng dẫn sử dụng
usage() {
    echo "Sử dụng: $0 [options]"
    echo "Options:"
    echo "  --dataset <name>         Tên dataset (vd: splitmnist, permmnist). Mặc định: $DATASET"
    echo "  --methods <list>         Danh sách phương pháp cách nhau bởi dấu phẩy (vd: uniform,coreset). Mặc định: uniform,kmeans_features,sensitivity,glister,coreset"
    echo "  --seeds <list>           Danh sách seed cách nhau bởi dấu phẩy (vd: 0,1,2). Mặc định: 0,1,2"
    echo "  --buffer-sizes <list>    Danh sách buffer size cách nhau bởi dấu phẩy. Mặc định: 50,100,200"
    echo "  --betas <list>           Danh sách beta cách nhau bởi dấu phẩy. Mặc định: 0.01,1.0,100.0"
    echo "  --stage <all|train|report>  Giai đoạn cần chạy. Mặc định: all"
    echo "                              - train:  chỉ chạy huấn luyện (cần GPU, torch, jax) và ghi cl_results/*.txt"
    echo "                              - report: chỉ tổng hợp bảng số liệu + vẽ biểu đồ từ cl_results/*.txt có sẵn"
    echo "                                        (chỉ cần numpy/matplotlib/seaborn, KHÔNG cần GPU/torch/jax)"
    echo "                              - all:    chạy cả train và report (mặc định, tương thích ngược)"
    echo "  --help                   Hiển thị trợ giúp này"
    exit 1
}

# Đọc tham số từ dòng lệnh
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --methods) IFS=',' read -r -a METHODS <<< "$2"; shift ;;
        --seeds) IFS=',' read -r -a SEEDS <<< "$2"; shift ;;
        --buffer-sizes) IFS=',' read -r -a BUFFER_SIZES <<< "$2"; shift ;;
        --betas) IFS=',' read -r -a BETAS <<< "$2"; shift ;;
        --stage) STAGE="$2"; shift ;;
        --help) usage ;;
        *) echo "Tham số không hợp lệ: $1"; usage ;;
    esac
    shift
done

if [[ "$STAGE" != "all" && "$STAGE" != "train" && "$STAGE" != "report" ]]; then
    echo "Giá trị --stage không hợp lệ: $STAGE (chỉ nhận all|train|report)"
    usage
fi

METHODS_CSV=$(IFS=,; echo "${METHODS[*]}")
SEEDS_CSV=$(IFS=,; echo "${SEEDS[*]}")
BETAS_CSV=$(IFS=,; echo "${BETAS[*]}")

# Chuyển vào thư mục chứa code
cd cl_streaming || exit
export PYTHONPATH="$(pwd)/..:$PYTHONPATH"

echo "================================================================"
echo "CẤU HÌNH THỰC NGHIỆM:"
echo "Dataset: $DATASET"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Buffer sizes (Ablation): ${BUFFER_SIZES[*]}"
echo "Betas (Ablation): ${BETAS[*]}"
echo "Stage: $STAGE"
echo "================================================================"
echo ""

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    echo "BẮT ĐẦU CHẠY THỰC NGHIỆM..."
    echo ""

    # --------------------------------------------------------------------------
    # 1. THỰC NGHIỆM CHÍNH (So sánh các phương pháp)
    # --------------------------------------------------------------------------
    echo "=== 1. KHẢO SÁT CÁC PHƯƠNG PHÁP ==="
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[INFO] Running: dataset=$DATASET | method=$method | seed=$seed | buffer_size=100 | beta=1.0"
            python cl.py --dataset "$DATASET" --method "$method" --seed "$seed" --buffer_size 100 --beta 1.0
        done
    done

    # --------------------------------------------------------------------------
    # 2. ABLATION STUDY 1: Ảnh hưởng của kích thước bộ nhớ (Buffer Size)
    # --------------------------------------------------------------------------
    echo "=== 2. ABLATION STUDY: KHẢO SÁT BUFFER SIZE (Phương pháp Coreset) ==="
    for b_size in "${BUFFER_SIZES[@]}"; do
        echo "[INFO] Running: dataset=$DATASET | method=coreset | seed=0 | buffer_size=$b_size | beta=1.0"
        python cl.py --dataset "$DATASET" --method coreset --seed 0 --buffer_size "$b_size" --beta 1.0
    done

    # --------------------------------------------------------------------------
    # 3. ABLATION STUDY 2: Ảnh hưởng của hệ số phạt (Beta)
    # --------------------------------------------------------------------------
    echo "=== 3. ABLATION STUDY: KHẢO SÁT BETA (Phương pháp Coreset) ==="
    for beta in "${BETAS[@]}"; do
        echo "[INFO] Running: dataset=$DATASET | method=coreset | seed=0 | buffer_size=100 | beta=$beta"
        python cl.py --dataset "$DATASET" --method coreset --seed 0 --buffer_size 100 --beta "$beta"
    done
    echo "Ghi chú: Kết quả chi tiết từng file được lưu trong thư mục cl_streaming/cl_results/"
    echo "Bạn có thể tải thư mục này về, rồi chạy lại script với '--stage report' để chỉ vẽ biểu đồ mà không cần huấn luyện lại."
else
    echo "Bỏ qua giai đoạn huấn luyện (--stage report). Dùng kết quả có sẵn trong cl_streaming/cl_results/."
fi

if [[ "$STAGE" == "all" || "$STAGE" == "report" ]]; then
    # --------------------------------------------------------------------------
    # 4. TỔNG HỢP VÀ IN KẾT QUẢ
    # --------------------------------------------------------------------------
    echo "=== 4. TỔNG HỢP KẾT QUẢ ==="
    python process_results.py --exp cl --datasets "$DATASET" --methods "$METHODS_CSV" --seeds "$SEEDS_CSV" --betas "$BETAS_CSV" --buffer_size 100

    # --------------------------------------------------------------------------
    # 5. TRỰC QUAN HÓA KẾT QUẢ
    # --------------------------------------------------------------------------
    echo "=== 5. TRỰC QUAN HÓA: FORGETTING ==="
    python ../experiments/tradeoff_time_accuracy_plot.py --plot forgetting

    echo "=== 6. TRỰC QUAN HÓA: TRADE-OFF THỜI GIAN vs ĐỘ CHÍNH XÁC ==="
    python ../experiments/tradeoff_time_accuracy_plot.py --plot tradeoff

    echo "=== 7. TRỰC QUAN HÓA: SO SÁNH VỚI BASELINE (Sensitivity Coreset, GLISTER) ==="
    python ../experiments/baseline_comparison_plot.py --plot all --dataset "$DATASET" --seeds "$SEEDS_CSV"
else
    echo "Bỏ qua giai đoạn tổng hợp/vẽ biểu đồ (--stage train). Chạy lại với '--stage report' sau khi đã có đủ cl_results/*.txt."
fi
