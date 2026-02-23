import cv2
import os


def preprocess_image(
    image_path: str,
    blur_kernel: tuple = (5, 5),
    canny_low: int = 50,
    canny_high: int = 150
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # NEW: Apply ROI
    edges = apply_roi_mask(edges)

    return edges


import numpy as np

def apply_roi_mask(edges):
    """
    Apply a trapezoidal region of interest mask
    to keep only the road area.
    """

    height, width = edges.shape

    # Define a trapezoid for road region
    roi_vertices = np.array([[
        (0, height),
        (width * 0.2, height * 0.6),
        (width * 0.8, height * 0.6),
        (width, height)
    ]], dtype=np.int32)

    # Create mask
    mask = np.zeros_like(edges)

    # Fill ROI polygon with white
    cv2.fillPoly(mask, roi_vertices, 255)

    # Apply mask
    masked_edges = cv2.bitwise_and(edges, mask)

    return masked_edges

import numpy as np
import cv2
def visualize_roi(image):
    height, width = image.shape[:2]

    roi_vertices = np.array([[
        (0, height),
        (width * 0.1, height * 0.75),
        (width * 0.9, height * 0.75),
        (width, height)
    ]], dtype=np.int32)

    overlay = image.copy()
    cv2.polylines(overlay, roi_vertices, True, (0, 255, 0), 3)

    return overlay
    
    def printu():
        print("The commit made today is for consistency")