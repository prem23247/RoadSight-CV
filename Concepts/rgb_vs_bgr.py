import cv2
import matplotlib.pyplot as plt

image = cv2.imread("data/raw/road.jpg")

# Direct display (BGR) – WRONG COLORS
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("BGR Image (Wrong Colors)")
plt.imshow(image)
plt.axis("off")

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.subplot(1,2,2)
plt.title("RGB Image (Correct Colors)")
plt.imshow(image_rgb)
plt.axis("off")

plt.show()
