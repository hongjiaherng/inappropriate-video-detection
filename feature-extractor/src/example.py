import numpy as np


batch_size_16_2 = np.load(
    "C:/Users/Jia Herng/Documents/Jia Herng's Docs/Final Year Project/inappropriate-video-detection/feature-extractor/data/outputs/i3d_rgb/1-1004/A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A_2.npy"
)

batch_size_16_1 = np.load(
    "C:/Users/Jia Herng/Desktop/A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A_1.npy"
)

batch_size_4 = np.load(
    "C:/Users/Jia Herng/Documents/Jia Herng's Docs/Final Year Project/inappropriate-video-detection/feature-extractor/data/outputs/i3d_rgb/1-1004/A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.npy"
)

print(batch_size_16_1.shape, batch_size_16_2.shape, batch_size_4.shape)
print(np.all(batch_size_16_1 == batch_size_16_2))
print(np.all(batch_size_16_1 == batch_size_4))
print(np.all(batch_size_16_2 == batch_size_4))
print(batch_size_16_1.mean())
print(batch_size_16_2.mean())
print(batch_size_4.mean())
print(batch_size_16_1.std())
print(batch_size_16_2.std())
print(batch_size_4.std())
