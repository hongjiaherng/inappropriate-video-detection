import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import HfFileSystem

fs = HfFileSystem()

file_infos = []
all_dirs = ["test_videos", "1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954"]
for dir in all_dirs:
    hf_dir_path = f"datasets/jherng/xd-violence/data/c3d_rgb/{dir}"

    for i, x in enumerate(fs.ls(hf_dir_path, detail=True)):
        file_infos.append(
            {
                "name": x["name"].split("datasets/jherng/xd-violence/data/c3d_rgb/")[-1].split(".np4")[0],
                "size": round(x["size"] / 1000000, 2),
            }
        )

df = pd.DataFrame(file_infos)
# Sort the DataFrame by the 'size' column in descending order
df_sorted = df.sort_values(by="size", ascending=False)
df_sorted.to_csv("c3d_feature_size.csv", sep=",", index=False)

# plt.figure(figsize=(10, 6))
# sns.histplot(df["size"], bins=500, kde=True, color="skyblue", edgecolor="black")  # You can adjust the number of bins as needed
# plt.title("Test Video Size Distribution")
# plt.xticks(range(0, int(df_sorted["size"].max()) + 1, 50))
# plt.xlabel("Size (in MB)")
# plt.ylabel("Frequency")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig("results/test_video_size_distribution.png")

# # Create a box plot
# plt.figure(figsize=(12, 6))
# sns.boxplot(x=df_sorted["size"], color="skyblue")
# plt.xticks(range(0, int(df_sorted["size"].max()) + 1, 50))
# plt.title("Box Plot of Test Video Size Distribution")
# plt.xlabel("Size (in MB)")
# plt.savefig("results/test_video_size_boxplot.png")
