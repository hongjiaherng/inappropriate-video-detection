# Download model from: https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html#i3d

target_directory="pretrained/"

# Create the directory if it doesn't exist
mkdir -p "$target_directory"

# Download the file
curl -O https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth

# Move the downloaded file to the desired directory
mv i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth "$target_directory"
