import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/raw/road.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresholds = [
    (30, 100),
    (50, 150),
    (100, 200),
    (150, 250)
]

plt.figure(figsize=(12, 8))

for i, (low, high) in enumerate(thresholds):
    edges = cv2.Canny(blur, low, high)
    plt.subplot(2, 2, i + 1)
    plt.imshow(edges, cmap="gray")
    plt.title(f"Canny: low={low}, high={high}")
    plt.axis("off")

plt.suptitle("Effect of Canny Thresholds on Edge Detection")
plt.tight_layout()
plt.show()