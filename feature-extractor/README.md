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
