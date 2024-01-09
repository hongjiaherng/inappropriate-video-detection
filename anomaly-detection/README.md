# anomaly-detection

## Usage

```bash
git clone https://github.com/hongjiaherng/inappropriate-video-detection.git

cd inappropriate-video-detection/anomaly-detection

chmod +x ./scripts/modify_hf_cache_path.sh
. ./scripts/modify_hf_cache_path.sh

pip install -r requirements.txt

wandb login <wandb-api-key>

# Sultani-Net
python src/main.py --config_path configs/sultani_net/sultaninet-swin.yaml sultani_net
python src/main.py --config_path configs/sultani_net/sultaninet-i3d.yaml sultani_net
python src/main.py --config_path configs/sultani_net/sultaninet-c3d.yaml sultani_net

# PengWu-Net (HLC context length = 1)
python src/main.py --config_path configs/pengwu_net/hlnet-ctx_len_1-swin.yaml pengwu_net
python src/main.py --config_path configs/pengwu_net/hlnet-ctx_len_1-i3d.yaml pengwu_net
python src/main.py --config_path configs/pengwu_net/hlnet-ctx_len_1-c3d.yaml pengwu_net

# PengWu-Net (HLC context length = 5)
python src/main.py --config_path configs/pengwu_net/hlnet-ctx_len_5-swin.yaml pengwu_net
python src/main.py --config_path configs/pengwu_net/hlnet-ctx_len_5-i3d.yaml pengwu_net
python src/main.py --config_path configs/pengwu_net/hlnet-ctx_len_5-c3d.yaml pengwu_net

# SVM Baseline
python src/main.py --config_path configs/svm_baseline/baseline-swin.yaml svm_baseline
python src/main.py --config_path configs/svm_baseline/baseline-i3d.yaml svm_baseline
python src/main.py --config_path configs/svm_baseline/baseline-c3d.yaml svm_baseline

```

## Model Training

| Model                        | Status (✓/x) |
| ---------------------------- | ------------ |
| Sultani-Net (swin)           | ✓            |
| Sultani-Net (i3d)            | ✓            |
| Sultani-Net (c3d)            | x            |
| PengWu-Net (ctx_len_1, swin) | x            |
| PengWu-Net (ctx_len_1, i3d)  | x            |
| PengWu-Net (ctx_len_1, c3d)  | x            |
| PengWu-Net (ctx_len_5, swin) | x            |
| PengWu-Net (ctx_len_5, i3d)  | x            |
| PengWu-Net (ctx_len_5, c3d)  | x            |
| SVM Baseline (swin)          | x            |
| SVM Baseline (i3d)           | x            |
| SVM Baseline (c3d)           | x            |
