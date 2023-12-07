#!/bin/bash
# Must set $HF_TOKEN in advance

# Run the commands in a loop (approximately 4k+ to finish all videos)
for i in {1..40}; do
    echo "Running iteration $i"
    python src/main.py \
        --batch_size=32 \
        --hf_dataset=jherng/xd-violence \
        --output_dir=data/outputs/i3d_rgb \
        --model=i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb \
        --num_clips_per_video=-1 \
        --crop_type=5-crop \
        --force_stop_after_n_videos=100 \
        --check_progress=data/outputs/progress.txt

    python src/upload_to_hf.py \
        --hf_dataset=jherng/xd-violence \
        --feature_dir=data/outputs/i3d_rgb \
        --path_in_repo=data/i3d_rgb \
        --remove_after_uploading \
        --hf_token=$HF_TOKEN

done

echo "Running final iteration to clear up incomplete videos"
python src/main.py \
    --batch_size=32 \
    --hf_dataset=jherng/xd-violence \
    --output_dir=data/outputs/i3d_rgb \
    --model=i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb \
    --num_clips_per_video=-1 \
    --crop_type=5-crop \
    --check_progress=data/outputs/progress.txt

python src/upload_to_hf.py \
    --hf_dataset=jherng/xd-violence \
    --feature_dir=data/outputs/i3d_rgb \
    --path_in_repo=data/i3d_rgb \
    --remove_after_uploading \
    --hf_token=$HF_TOKEN

echo "Script completed."
