from huggingface_hub import HfFileSystem

fs = HfFileSystem()

i3d_rgb_list = fs.glob("datasets/jherng/xd-violence/data/i3d_rgb/**.npy")
i3d_rgb_list = [
    x.split("datasets/jherng/xd-violence/data/i3d_rgb/")[-1].split(".npy")[0]
    for x in i3d_rgb_list
]

with open("results/progress.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(i3d_rgb_list))

i3d_rgb_list = fs.glob("datasets/jherng/xd-violence/data/video/**.mp4")
i3d_rgb_list = [
    x.split("datasets/jherng/xd-violence/data/video/")[-1].split(".mp4")[0]
    for x in i3d_rgb_list
]


with open("results/videos.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(i3d_rgb_list))
