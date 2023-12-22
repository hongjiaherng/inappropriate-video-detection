## video-feature-extractor

CLI to extract clip/snippet features from videos using I3D ResNet-50 and Video Swin Transformer backbones. This repository is primarily integrated with my [huggingface repository for XD-Violence dataset](https://huggingface.co/datasets/jherng/xd-violence) to extract features for the dataset. Extracting features from videos located at arbitrary local paths is also supported.

## Usage

### Setup

1. **Install Dependencies**

```bash
# Install PyTorch and Hugging Face datasets
pip install -r requirements.txt

# Install mmaction2
pip install -U openmim
mim install mmengine
mim install mmcv
pip install mmaction2

# Or with my own utility script (add the missing folders associated with mmaction2)
chmod +x ./scripts/install_mmaction2.sh
./scripts/install_mmaction2.sh
```

2. **Download Pretrained Models**

- The pretrained models are downloaded from mmaction2's [model zoo](https://mmaction2.readthedocs.io/en/latest/modelzoo_statistics.html).

```bash
# Download I3D ResNet-50 model
curl -O https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth
mv i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth pretrained/

# Download Video Swin Transformer model
curl -O https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth
mv swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth pretrained/

# Or with my own utility script
chmod +x ./scripts/download_models.sh
./scripts/download_models.sh
```

### Arguments

| Argument                      | Description                                                                                                                                                                                                                                                                 | Choices and Defaults                                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--hf_dataset`                | HuggingFace dataset name containing source videos for visual feature extraction. Mutually exclusive with `--input_dir`.                                                                                                                                                     | Only supports `["jherng/xd-violence"]`                                                                                                                                                                                                                     |
| `--input_dir`                 | Directory for input videos to extract visual features locally. Mutually exclusive with `--hf_dataset`.                                                                                                                                                                      | -                                                                                                                                                                                                                                                          |
| `--output_dir`                | Directory to store extracted features.                                                                                                                                                                                                                                      | Default: `./data/outputs/feature/`                                                                                                                                                                                                                         |
| `--batch_size`                | Batch size for processing clips per video. Each batch of clips from a video is processed sequentially, with results concatenated. If -1, all the clips from a video is entirely loaded into memory at once, this might cause out-of-memory issue if the video is very long. | Default: `16`                                                                                                                                                                                                                                              |
| `--model`                     | Model for feature extraction.                                                                                                                                                                                                                                               | Default: `"i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"`<br>Choices: `["i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb", "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb"]` |
| `--num_clips_per_video`       | Number of clips to sample from each video. If -1, extracts all possible clips from start to end. Must satisfy (== -1 \|\| >= 1).                                                                                                                                            | Default: `-1`                                                                                                                                                                                                                                              |
| `--crop_type`                 | Cropping technique for spatially cutting input videos into n cropped videos of size 224x224 before feature extraction.                                                                                                                                                      | Default: `"5-crop"`<br>Choices: `["10-crop", "5-crop", "center"]`                                                                                                                                                                                          |
| `--force_stop_after_n_videos` | Maximum number of videos to extract features from. If -1, extracts features from all available videos exhaustively.                                                                                                                                                         | Default: `-1`                                                                                                                                                                                                                                              |
| `--check_progress`            | Progress determination for feature extraction. If None, check `<output_dir>` to determine progress; if `<filename>`, check line by line.                                                                                                                                    | Default: `None`                                                                                                                                                                                                                                            |

### Run

Extract features from XD-Violence dataset using I3D ResNet-50 backbone with 5-crop spatial cropping technique. The extracted features are stored in `data/outputs/i3d_rgb/` and the progress is tracked in `results/progress.txt`.

```bash
python ./src/main.py \
    --batch_size=32 \
    --hf_dataset=jherng/xd-violence \
    --output_dir=./data/outputs/i3d_rgb \
    --model=i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb \
    --num_clips_per_video=-1 \
    --crop_type=5-crop \
    --force_stop_after_n_videos=-1 \
    --check_progress=./results/progress.txt
```

Extract features from local directory `data/inputs/` using Video Swin Transformer backbone with 10-crop spatial cropping technique. The extracted features are stored in `data/outputs/swin_rgb/`.

```bash
python ./src/main.py \
    --batch_size=32 \
    --input_dir=./data/inputs/ \
    --output_dir=./data/outputs/swin_rgb \
    --model=swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb \
    --num_clips_per_video=-1 \
    --crop_type=10-crop \
    --force_stop_after_n_videos=-1 \
```

## Notes

- The preprocessing pipeline for videos is dependent on the pretrained model. The preprocessing pipeline and configuration for each model can be found in `src/models/`.
  - The preprocessing pipeline looks generally like this:
    ```
    Initialize video reader ➔ Sample frame indices for clips ➔ Organize frame indices into batches (if applicable) ➔ Load clips ➔ Resize clips ➔ Crop clips ➔ Normalize clips based on ImageNet's mean and standard deviation ➔ Organize clips' dimensions as (batch_size, num_crops, num_channels, num_frames, height, width) or (B, n_crops, C, T, H, W) ➔ Ready for feature extraction
    ```
