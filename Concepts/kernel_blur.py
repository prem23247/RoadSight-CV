import cv2
import matplotlib.pyplot as plt

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread("data/raw/road.jpg")
if img is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Different kernel sizes to test
kernel_sizes = [3, 5, 9, 15]

plt.figure(figsize=(14, 8))

for i, k in enumerate(kernel_sizes):
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Display
    plt.subplot(2, 2, i + 1)
    plt.imshow(edges, cmap="gray")
    plt.title(f"Kernel Size: {k}x{k}")
    plt.axis("off")

plt.suptitle("Effect of Gaussian Kernel Size on Edge Detection", fontsize=14)
plt.tight_layout()
plt.show()