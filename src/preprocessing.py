import cv2
import os


def preprocess_image(
    image_path: str,
    blur_kernel: tuple = (5, 5),
    canny_low: int = 50,
    canny_high: int = 150
):
    """
    Preprocess a road image for edge detection.

    Steps:
    1. Read image (BGR)
    2. Convert to grayscale
    3. Apply Gaussian blur
    4. Apply Canny edge detection

    Returns:
        edges (numpy.ndarray): Binary edge image
    """

    # 1. Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian Blur
    blur = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 4. Canny Edge Detection
    edges = cv2.Canny(blur, canny_low, canny_high)

    return edges