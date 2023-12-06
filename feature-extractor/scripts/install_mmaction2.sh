pip install -U openmim
mim install mmengine
mim install mmcv
pip install mmaction2

# fix the modulenotfounderror in mmaction2
# %cd /kaggle/working/inappropriate-video-detection/feature-extractor
# !unzip scripts/drn.zip
# !ls scripts/drn/
# !mv scripts/drn/ /opt/conda/lib/python3.10/site-packages/mmaction/models/localizers/
# !ls /opt/conda/lib/python3.10/site-packages/mmaction/models/localizers