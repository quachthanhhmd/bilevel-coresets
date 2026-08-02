# Coresets via Bilevel Optimization

Đây là cài đặt tham chiếu cho bài báo "Data Summarization via Bilevel Optimization"
([arXiv:2109.12534](https://arxiv.org/abs/2109.12534)).

## Yêu cầu môi trường

Cần Python 3. Cài các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

## Cấu trúc mã nguồn

| Thư mục / file | Vai trò |
| --- | --- |
| `bilevel_coreset.py`, `loss_utils.py`, `models.py`, `jax_patch.py` | Thư viện gốc (bản hội nghị) — coreset qua proxy, kiến trúc model, các loss dùng chung |
| `bicoreset/` |  Xây coreset cho Algorithm 1 |
| `cl_streaming/` | thực nghiệm Continual Learning |
| `data_summarization/` | Thực nghiệm tóm tắt dữ liệu  |
| `experiments/` | Các script cần dùng để chạy thực nghiệm và visualize các thực nghiệm đó |
| `experiments/results` | Các script cần dùng để chạy thực nghiệm và visualize các thực nghiệm đó |
| `demos/` | Ví dụ ngắn minh hoạ cách dùng `bicoreset/` cho từng bài toán (không phải thực nghiệm) |
| `tests/` | Bộ test `pytest` kiểm chứng cài đặt theo đúng công thức toán |
| `algorithm1-cifar10-experiment.ipynb`, `all-experiments.ipynb`, `kmist.ipynb` | Notebook Kaggle/Colab chạy sẵn một hoặc nhiều thực nghiệm |

## Cách chạy các thực nghiệm


Để dễ dàng chạy và không cần cài đặt môi trường, có thể dùng file
`fashionkmnist.ipynb` để chạy thực nghiệm cho fashionkmnist và `kmist.ipynb` để chạy thực nghiệm cho kmist


Vì mối thuật toán chạy rất lâu, để chạy riêng cho từng thực nghiêm, chúng ta có thể dụng các lệch sau

### 1. Algorithm 1

```bash
./run_algorithm1_paper_experiments.sh --datasets cifar10,fashionmnist
# hoặc gọi trực tiếp từng bước:
python experiments/run_algorithm1_variants_paper_cifar10.py --dataset cifar10 --method bico_fwd --size-pct 8 --seed 0
python experiments/algorithm1_variants_paper_cifar10_plot.py --dataset cifar10
```

Notebook tương ứng: `algorithm1-cifar10-experiment.ipynb`.

### 2. Hồi quy logistic + coreset

```bash
python demos/demo_logistic_regression.py --dataset sklearn 
python demos/demo_logistic_regression.py --dataset kmnist    
```

### 3. GMM (Gaussian Mixture Model)

```bash
python experiments/run_gmm_paper.py --dataset synthetic            
python experiments/run_gmm_paper.py --dataset kmnist            
./experiments/run_gmm_multi_dataset.sh --datasets kmnist,cifar10,fashionmnist
```

### 4. Continual Learning

Một lần chạy cho một `(dataset, method, seed)`:

```bash
cd cl_streaming
python cl.py --dataset splitfashionmnist --method coreset --seed 0 --buffer_size 100 --beta 1.0
```

Chạy toàn bộ khảo sát method + vẽ biểu đồ tổng hợp bằng một lệnh:

```bash
./run_experiments.sh --dataset splitfashionmnist --stage all
```

`--stage train` chỉ huấn luyện (ghi `cl_streaming/cl_results/`), `--stage report` chỉ
tổng hợp bảng số liệu và vẽ biểu đồ từ kết quả có sẵn (không cần GPU/torch/jax).


### 5. Ablation Study

Ablation (buffer size và beta) nằm trong cùng `run_experiments.sh` ở mục 4 — mặc định
quét `buffer_size` = `50,100,200` và `beta` = `0.01,1.0,100.0` cho method `coreset`.
Tuỳ chỉnh:

```bash
./run_experiments.sh --dataset splitfashionmnist --buffer-sizes 50,100,200 --betas 0.01,1.0,100.0
```

Vẽ riêng biểu đồ ablation sau khi đã có kết quả:

```bash
python experiments/baseline_comparison_plot.py --plot ablation --dataset splitfashionmnist --ablation_method coreset --ablation_seed 0
```
