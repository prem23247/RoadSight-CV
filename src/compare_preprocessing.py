import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import preprocess_image


INPUT_DIR = "data/raw"

for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)

    # Read original image
    original = cv2.imread(image_path)
    if original is None:
        continue

    # Convert BGR → RGB for correct display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Get edge output
    edges = preprocess_image(image_path)

    # Show side-by-side comparison
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Edge Output (Canny)")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.suptitle(image_name)
    plt.tight_layout()
    plt.show()