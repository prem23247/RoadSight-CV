import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_image, apply_roi_mask, visualize_roi


INPUT_DIR = "data/raw"


for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)

    # -----------------------------
    # Read original image
    # -----------------------------
    original = cv2.imread(image_path)
    if original is None:
        continue

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Visualize ROI on original image
    # -----------------------------
    image_with_roi = visualize_roi(original_rgb)

    plt.figure(figsize=(6, 6))
    plt.title("Original Image with ROI")
    plt.imshow(image_with_roi)
    plt.axis("off")
    plt.show()

    # -----------------------------
    # Edges WITHOUT ROI (baseline)
    # -----------------------------
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_no_roi = cv2.Canny(blur, 50, 150)

    # -----------------------------
    # Edges WITH ROI (project pipeline)
    # -----------------------------
    edges_with_roi = apply_roi_mask(edges_no_roi)

    # -----------------------------
    # Display comparison
    # -----------------------------
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Edges (No ROI)")
    plt.imshow(edges_no_roi, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Edges (With ROI)")
    plt.imshow(edges_with_roi, cmap="gray")
    plt.axis("off")

    plt.suptitle(image_name)
    plt.tight_layout()
    plt.show()

    