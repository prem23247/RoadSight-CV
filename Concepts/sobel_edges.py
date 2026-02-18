import cv2
import matplotlib.pyplot as plt

image = cv2.imread("data/raw/road.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Sobel X")
plt.imshow(sobel_x, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Sobel Y")
plt.imshow(sobel_y, cmap="gray")
plt.axis("off")

plt.show()
