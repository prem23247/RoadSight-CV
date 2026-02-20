import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/raw/road.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

configs = [
    ((5, 5), 0),
    ((5, 5), 1),
    ((5, 5), 3),
    ((9, 9), 1),
]

plt.figure(figsize=(12, 8))

for i, (ksize, sigma) in enumerate(configs):
    blur = cv2.GaussianBlur(gray, ksize, sigma)
    edges = cv2.Canny(blur, 50, 150)

    plt.subplot(2, 2, i + 1)
    plt.imshow(edges, cmap="gray")
    plt.title(f"Kernel={ksize}, Sigma={sigma}")
    plt.axis("off")

plt.suptitle("Effect of Sigma vs Kernel Size")
plt.tight_layout()
plt.show()