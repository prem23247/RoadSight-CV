import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Read Image (BGR)
# -----------------------------
img = cv2.imread("data/raw/road.jpg")

if img is None:
    raise FileNotFoundError("Image not found")

# Convert for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------
# 2. Grayscale
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Gaussian Blur
# -----------------------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -----------------------------
# 4. Sobel Edges
# -----------------------------
sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# -----------------------------
# 5. Canny Edges
# -----------------------------
canny = cv2.Canny(blur, 50, 150)

# -----------------------------
# 6. Display Everything Together
# -----------------------------
plt.figure(figsize=(14, 6))

plt.subplot(1, 5, 1)
plt.title("Original (RGB)")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 5, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.title("Gaussian Blur")
plt.imshow(blur, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.title("Sobel Edges")
plt.imshow(sobel, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.title("Canny Edges")
plt.imshow(canny, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()