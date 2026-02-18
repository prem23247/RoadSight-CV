import cv2

image = cv2.imread("data/raw/road.jpg")

if image is None:
    print("ERROR: Image not loaded. Check file path or name.")
else:
    print("Image loaded successfully")
    print("Image type:", type(image))
    print("Image shape:", image.shape)
