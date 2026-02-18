import cv2
import matplotlib.pyplot as plt

image = cv2.imread("data/raw/road.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original (RGB)")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.show()
