import cv2
import numpy as np
import os

def harris_corner_detection(reference_image_path: str, output_path: str):
    """
    Loads an image, runs Harris corner detection, and saves an output image
    with the detected corners marked in red.

    reference_image_path: path to input image (reference_img.png)
    output_path: where to save the result (output/harris.png)
    """

    # 1. Read image
    img = cv2.imread(reference_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {reference_image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Convert to float32 (required by cv2.cornerHarris)
    gray_float = np.float32(gray)

    # 3. Harris corner detection
    # blockSize: neighborhood size for corner detection
    # ksize: aperture parameter of the Sobel derivative
    # k: Harris detector free parameter (0.04-0.06 typical)
    dst = cv2.cornerHarris(src=gray_float, blockSize=2, ksize=3, k=0.04)

    # 4. Dilate result for marking (makes the corners more visible)
    dst = cv2.dilate(dst, None)

    # 5. Threshold and mark detected corners in red
    img_marked = img.copy()
    img_marked[dst > 0.01 * dst.max()] = [0, 0, 255]  # BGR -> red

    # 6. Make sure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 7. Save
    cv2.imwrite(output_path, img_marked)
    print(f"[OK] Harris corners saved to {output_path}")


if __name__ == "__main__":
    reference_image_path = "images/reference_img.png"
    output_path = "output/harris.png"
    harris_corner_detection(reference_image_path, output_path)
