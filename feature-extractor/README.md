## Starting up in a Kaggle Notebook / Google Colab

```bash
!git clone https://github.com/hongjiaherng/inappropriate-video-detection.git
%cd inappropriate-video-detection/feature-extractor
!pip install -r requirements.txt

!chmod +x scripts/install_mmaction2.sh
!./scripts/install_mmaction2.sh

!chmod +x scripts/download_model.sh
!./scripts/download_model.sh
```

## Starting up in your own PC

```bash
conda create --name <env-name> python=3.10
pip install -r requirements.txt
sh scripts/install_mmaction2.sh
sh scripts/download_model.sh
```


This video is one sec (can't extract feature)
- 2005-2804/v=Gm73TwtUyGY__#1_label_G-0-0.mp4 (https://huggingface.co/datasets/jherng/xd-violence/blob/main/data/video/2005-2804/v%3DGm73TwtUyGY__%231_label_G-0-0.mp4)
