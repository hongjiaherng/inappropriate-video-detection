#!/bin/bash

# Run the commands in a loop (approximately 4k+ to finish all videos)
for i in {1..40}; do
    echo "Running iteration $i"
    python src/main.py \
        --hf_dataset=jherng/xd-violence \
        --output_dir=data/outputs/i3d_rgb \
        --model=i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb \
        --num_clips_per_video=-1 \
        --crop_type=5-crop \
        --force_stop_after_n_videos=100

    python src/upload_to_hf.py \
        --hf_dataset=jherng/xd-violence \
        --feature_dir=data/outputs/i3d_rgb \
        --path_in_repo=data/i3d_rgb
done

echo "Running final iteration to clear up incomplete videos"
python src/main.py \
    --hf_dataset=jherng/xd-violence \
    --output_dir=data/outputs/i3d_rgb \
    --model=i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb \
    --num_clips_per_video=-1 \
    --crop_type=5-crop

python src/upload_to_hf.py \
    --hf_dataset=jherng/xd-violence \
    --feature_dir=data/outputs/i3d_rgb \
    --path_in_repo=data/i3d_rgb

echo "Script completed."
