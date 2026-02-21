import os
import cv2
from preprocessing import preprocess_image


INPUT_DIR = "data/raw"
OUTPUT_DIR = "results/edges"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)

    edges = preprocess_image(image_path)

    output_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(output_path, edges)

    print(f"[OK] Processed: {image_name}")