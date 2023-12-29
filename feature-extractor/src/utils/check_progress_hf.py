from huggingface_hub import HfFileSystem

feat_name = "swin_rgb"
fs = HfFileSystem()

rgb_list = fs.glob(f"datasets/jherng/xd-violence/data/{feat_name}/**.npy")
rgb_list = [
    x.split(f"datasets/jherng/xd-violence/data/{feat_name}/")[-1].split(".npy")[0]
    for x in rgb_list
]

with open(f"results/{feat_name}_progress.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rgb_list))
    f.write("\n")

