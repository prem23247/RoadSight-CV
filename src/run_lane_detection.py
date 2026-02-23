import os
import cv2
import matplotlib.pyplot as plt

from src.preprocessing import (
    preprocess_image,
    detect_lane_lines,
    draw_lane_lines
)

INPUT_DIR = "data/raw"

for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)

    # Read original image
    original = cv2.imread(image_path)
    if original is None:
        continue

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Step 1: Edge detection (with ROI)
    edges = preprocess_image(image_path)

    # Step 2: Hough Transform
    lines = detect_lane_lines(edges)

    # Step 3: Draw lines
    lane_image = draw_lane_lines(original, lines)
    lane_image_rgb = cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Edges (ROI)")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Lane Detection (Hough)")
    plt.imshow(lane_image_rgb)
    plt.axis("off")

    plt.suptitle(image_name)
    plt.tight_layout()
    plt.show()