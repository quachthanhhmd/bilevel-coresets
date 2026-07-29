#!/bin/bash

# ==============================================================================
# Script chạy thực nghiệm Sec 5.1 / Figure 3 (CNTK-Nystrom + logistic regression)
# cho CIFAR-10 (bài báo gốc) và FashionMNIST (cùng pipeline, áp dụng sang dataset
# khác -- không phải số liệu paper, xem docstring của
# experiments/run_algorithm1_variants_paper_cifar10.py). Một lệnh duy nhất chạy
# cả build coreset (--stage train) lẫn vẽ biểu đồ (--stage report), để backup/tái
# tạo toàn bộ dữ liệu cần cho visualize mà không cần bấm qua nhiều cell notebook.
#
# Mặc định giữ đúng tham số Appendix C của paper cho cả 2 dataset (không tự động
# scale nhỏ) -- script chỉ là vòng lặp gọi run_algorithm1_variants_paper_cifar10.py,
# không đổi hyperparameter mặc định của nó.
# ==============================================================================

# Giá trị mặc định
DATASETS=("cifar10" "fashionmnist")
METHODS=("uniform" "bico_fwd" "bico_fwd25" "bico_elim" "bico_exch" "bico_reg")
SIZES_PCT=(0.5 2 8 32)
SEEDS=(0)
STAGE="all"

usage() {
    echo "Sử dụng: $0 [options]"
    echo "Options:"
    echo "  --datasets <list>   Danh sách dataset cách nhau bởi dấu phẩy (cifar10,fashionmnist). Mặc định: cifar10,fashionmnist"
    echo "  --methods <list>    Danh sách method cách nhau bởi dấu phẩy. Mặc định: uniform,bico_fwd,bico_fwd25,bico_elim,bico_exch,bico_reg"
    echo "  --sizes-pct <list>  Danh sách size (% train partition) cách nhau bởi dấu phẩy. Mặc định: 0.5,2,8,32"
    echo "  --seeds <list>      Danh sách seed cách nhau bởi dấu phẩy. Mặc định: 0"
    echo "  --stage <all|train|report>  Giai đoạn cần chạy. Mặc định: all"
    echo "                        - train:  chỉ build coreset + train, ghi experiments/algo1_paper_<dataset>_results/*.txt"
    echo "                        - report: chỉ vẽ biểu đồ từ kết quả có sẵn (algorithm1_variants_paper_cifar10_plot.py)"
    echo "                        - all:    chạy cả 2 (mặc định)"
    echo "  --help              Hiển thị trợ giúp này"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --datasets) IFS=',' read -r -a DATASETS <<< "$2"; shift ;;
        --methods) IFS=',' read -r -a METHODS <<< "$2"; shift ;;
        --sizes-pct) IFS=',' read -r -a SIZES_PCT <<< "$2"; shift ;;
        --seeds) IFS=',' read -r -a SEEDS <<< "$2"; shift ;;
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

SIZES_PCT_CSV=$(IFS=,; echo "${SIZES_PCT[*]}")
SEEDS_CSV=$(IFS=,; echo "${SEEDS[*]}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit

echo "================================================================"
echo "CẤU HÌNH THỰC NGHIỆM (Sec 5.1 / Figure 3, CNTK-Nystrom):"
echo "Datasets: ${DATASETS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Sizes (%): ${SIZES_PCT[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Stage: $STAGE"
echo "================================================================"
echo ""

for dataset in "${DATASETS[@]}"; do
    if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
        echo "=== [$dataset] Full Dataset (đường tham chiếu + tính/cache feature CNTK-Nystrom) ==="
        for seed in "${SEEDS[@]}"; do
            echo "[INFO] dataset=$dataset | method=full | seed=$seed"
            python experiments/run_algorithm1_variants_paper_cifar10.py \
                --dataset "$dataset" --method full --seed "$seed" || exit 1
        done

        echo "=== [$dataset] Các biến thể Algorithm 1 ==="
        for method in "${METHODS[@]}"; do
            for size_pct in "${SIZES_PCT[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    echo "[INFO] dataset=$dataset | method=$method | size_pct=$size_pct | seed=$seed"
                    python experiments/run_algorithm1_variants_paper_cifar10.py \
                        --dataset "$dataset" --method "$method" --size-pct "$size_pct" --seed "$seed" || exit 1
                done
            done
        done
    else
        echo "[$dataset] Bỏ qua giai đoạn train (--stage report)."
    fi

    if [[ "$STAGE" == "all" || "$STAGE" == "report" ]]; then
        echo "=== [$dataset] Vẽ biểu đồ ==="
        python experiments/algorithm1_variants_paper_cifar10_plot.py \
            --dataset "$dataset" --sizes-pct "$SIZES_PCT_CSV" --seeds "$SEEDS_CSV" || exit 1
    fi
    echo ""
done

echo "Xong. Kết quả từng dataset nằm trong experiments/algo1_paper_<dataset>_results/*.txt"
echo "(giữ nguyên thư mục này để backup -- không cần build lại feature CNTK-Nystrom nếu chạy lại)."
echo "Biểu đồ: experiments/algorithm1_variants_paper_<dataset>_accuracy.png"
