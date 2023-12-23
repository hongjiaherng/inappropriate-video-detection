from datasets import Features, Array3D

features = Features({"x": Array3D(shape=(1, 2, 3), dtype="int32")})

print(features)