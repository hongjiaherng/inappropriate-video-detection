def find_duplicates(mylist):
    newlist = []  # empty list to hold unique elements from the list
    duplist = []  # empty list to hold the duplicate elements from the list
    for i in mylist:
        if i not in newlist:
            newlist.append(i)
        else:
            duplist.append(i)
    return duplist


with open(
    "C:/Users/Jia Herng/Documents/Jia Herng's Docs/Final Year Project/inappropriate-video-detection/feature-extractor/skipped_videos.txt",
    "r",
) as f:
    lines = f.readlines()
    skipped = set([line.strip() for line in lines])

with open(
    "C:/Users/Jia Herng/Documents/Jia Herng's Docs/Final Year Project/inappropriate-video-detection/feature-extractor/hf_progress.txt",
    "r",
) as f:
    lines = f.readlines()
    hf_progress = set([line.strip() for line in lines])

with open(
    "C:/Users/Jia Herng/Documents/Jia Herng's Docs/Final Year Project/inappropriate-video-detection/feature-extractor/new_progress.txt",
    "r",
) as f:
    lines = f.readlines()
    new_progress = [line.strip() for line in lines]

dup = find_duplicates(new_progress)

skipped_plus_hf_progress = skipped.union(hf_progress)
leftover = skipped_plus_hf_progress.difference(new_progress)

print(dup)
print(f"{len(skipped)=}")
print(f"{len(hf_progress)=}")
print(f"{len(new_progress)=}")
print(f"{len(skipped_plus_hf_progress)=}")
print(f"{len(leftover)=}")
