import os

# Path to your dataset folder
dataset_dir = "data"
os.makedirs(dataset_dir, exist_ok=True)

# Content of README.md
readme_content = """# MNIST Dataset Documentation

## 📌 Overview
This dataset contains **handwritten digit images (0–9)** from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
Each image is **28x28 pixels (grayscale)** and labeled with its corresponding digit.  

It is commonly used for:
- Image classification
- Computer vision experiments
- Deep learning benchmarking

---

## 📂 Dataset Structure

---

## 🔎 Data Format
- **Images**: NumPy arrays, shape `(num_samples, 28, 28)`  
  - Values normalized to `[0, 1]` if preprocessed  
  - Otherwise stored as pixel values `[0, 255]`  

- **Labels**: NumPy arrays, shape `(num_samples,)`  
  - Integer values from `0` to `9`  

---

## 🚀 Usage Example (Python)
```python
import numpy as np

# Load dataset
x_train = np.load("data/train_images.npy")
y_train = np.load("data/train_labels.npy")
x_test  = np.load("data/test_images.npy")
y_test  = np.load("data/test_labels.npy")


---

⚡ This will:  
1. Create a `data/` folder (if it doesn’t exist).  
2. Write a **README.md** file with MNIST dataset documentation inside it.  

Do you also want me to **extend this script** so it **automatically generates `train_images.npy`, `train_labels.npy`, etc.** from Keras MNIST dataset along with the README?


print("Train shape:", x_train.shape, y_train.shape)
print("Test shape:", x_test.shape, y_test.shape)
