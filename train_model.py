from datasets import load_dataset
import numpy as np

print("Downloading CIFAR-10 from Hugging Face...")
ds = load_dataset("uoft-cs/cifar10")

print("Converting to numpy arrays...")
x_train = np.array([np.array(img) for img in ds["train"]["img"]])
y_train = np.array(ds["train"]["label"])
x_test = np.array([np.array(img) for img in ds["test"]["img"]])
y_test = np.array(ds["test"]["label"])

print("Train shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test shape:", x_test.shape)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print("Saved all arrays to disk as .npy files.")