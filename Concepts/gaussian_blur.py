import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("data/raw/road.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Display comparison
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("After Gaussian Blur")
plt.imshow(blur, cmap="gray")
plt.axis("off")

plt.show()
