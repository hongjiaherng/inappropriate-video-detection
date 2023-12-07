import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import HfFileSystem

fs = HfFileSystem()

file_infos = []
all_dirs = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
for dir in all_dirs:
    hf_dir_path = f"datasets/jherng/xd-violence/data/video/{dir}"

    for i, x in enumerate(fs.ls(hf_dir_path, detail=True)):
        file_infos.append(
            {
                "name": x["name"]
                .split("datasets/jherng/xd-violence/data/video/")[-1]
                .split(".mp4")[0],
                "size": round(x["size"] / 1000000, 2),
            }
        )

df = pd.DataFrame(file_infos)
# Sort the DataFrame by the 'size' column in descending order
df_sorted = df.sort_values(by="size", ascending=False)
df_sorted.to_csv("results/video_size_info.csv", sep=",", index=False)

plt.figure(figsize=(10, 6))
sns.histplot(
    df["size"], bins=500, kde=True, color="skyblue", edgecolor="black"
)  # You can adjust the number of bins as needed
plt.title("Video Size Distribution")
plt.xticks(range(0, int(df_sorted["size"].max()) + 1, 50))
plt.xlabel("Size (in MB)")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("results/video_size_distribution.png")

# Create a box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_sorted["size"], color="skyblue")
plt.xticks(range(0, int(df_sorted["size"].max()) + 1, 50))
plt.title("Box Plot of Video Size Distribution")
plt.xlabel("Size (in MB)")
plt.savefig("results/video_size_boxplot.png")

i3d_rgb_list = fs.glob("datasets/jherng/xd-violence/data/i3d_rgb/**.npy")
i3d_rgb_list = [
    x.split("datasets/jherng/xd-violence/data/i3d_rgb/")[-1].split(".npy")[0]
    for x in i3d_rgb_list
]

print(len(i3d_rgb_list))
print(i3d_rgb_list)

with open("results/progress.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(i3d_rgb_list))

i3d_rgb_list = fs.glob("datasets/jherng/xd-violence/data/video/**.mp4")
i3d_rgb_list = [
    x.split("datasets/jherng/xd-violence/data/video/")[-1].split(".mp4")[0]
    for x in i3d_rgb_list
]

print(len(i3d_rgb_list))
print(i3d_rgb_list)

with open("results/videos.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(i3d_rgb_list))
