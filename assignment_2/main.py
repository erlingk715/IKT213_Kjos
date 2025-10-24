# assignment_2/main.py
import cv2
import numpy as np
from pathlib import Path

# ---------- Utility ----------
def ensure_dirs():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "solutions"
    out_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, out_dir

def load_lena(script_dir: Path):
    # Prøv i script-mappa først, deretter i foreldremappa
    candidates = [
        script_dir / "lena.png",
        script_dir.parent / "lena.png",
    ]
    for p in candidates:
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                print(f"[Info] Loaded lena.png from: {p}")
                return img
    print("[Error] Could not find/load lena.png. Place it next to main.py.")
    return None

# ---------- I. FUNCTIONS (as required) ----------
def padding(image, border_width):
    """Reflective border around the image."""
    return cv2.copyMakeBorder(
        image, border_width, border_width, border_width, border_width,
        borderType=cv2.BORDER_REFLECT
    )

def crop(image, x_0, x_1, y_0, y_1):
    """Crop using pixel indices (x from x_0:x_1, y from y_0:y_1)."""
    return image[y_0:y_1, x_0:x_1]

def resize(image, width, height):
    """Resize to (width, height)."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def copy(image, emptyPictureArray):
    """Manual copy: fill target array pixel-by-pixel (no cv2.copy)."""
    h, w = image.shape[:2]
    for y in range(h):
        for x in range(w):
            emptyPictureArray[y, x] = image[y, x]
    return emptyPictureArray

def grayscale(image):
    """Convert BGR -> Grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image):
    """Convert BGR -> HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image, hue_degrees):
    """
    Ekte hue-shift: konverter BGR->HSV, forskyv kun H-kanalen.
    hue_degrees forventes i [0, 360). OpenCV bruker H i [0,179].
    """
    hue_degrees = float(hue_degrees) % 360.0
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    shift = int(hue_degrees * 180.0 / 360.0)  # grader -> [0,179]
    h = (h.astype(np.int16) + shift) % 180
    h = h.astype(np.uint8)
    hsv_shifted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

def smoothing(image):
    """Gaussian blur ksize=(15,15), default border."""
    return cv2.GaussianBlur(image, (15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT)

def rotation(image, rotation_angle):
    """Rotate by 90 (clockwise) or 180."""
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle must be 90 or 180")

# ---------- II. DEMO/OUTPUT (saves all required results) ----------
if __name__ == "__main__":
    script_dir, out_dir = ensure_dirs()
    img = load_lena(script_dir)
    if img is None:
        raise SystemExit(1)

    h, w, c = img.shape

    # Padding (reflect), border_width=100
    padded = padding(img, 100)
    cv2.imwrite(str(out_dir / "lena_padding_reflect_100.png"), padded)

    # Crop: remove 80 left/top, 130 right/bottom
    x0, y0 = 80, 80
    x1, y1 = w - 130, h - 130
    cropped = crop(img, x0, x1, y0, y1)
    cv2.imwrite(str(out_dir / "lena_cropped.png"), cropped)

    # Resize to 200x200
    resized = resize(img, 200, 200)
    cv2.imwrite(str(out_dir / "lena_resized_200x200.png"), resized)

    # Manual copy (no cv2.copy)
    emptyPictureArray = np.zeros((h, w, 3), dtype=np.uint8)
    copied = copy(img, emptyPictureArray)
    cv2.imwrite(str(out_dir / "lena_copied_manual.png"), copied)

    # Grayscale
    gray = grayscale(img)
    cv2.imwrite(str(out_dir / "lena_grayscale.png"), gray)

    # HSV (lagres som HSV-data – ser “falsk farge” ut i visere, det er ok)
    hsv_img = hsv(img)
    cv2.imwrite(str(out_dir / "lena_hsv.png"), hsv_img)

    # Ekte hue shift: +50 grader
    shifted = hue_shifted(img, 50)
    cv2.imwrite(str(out_dir / "lena_hue_shifted_plus50.png"), shifted)

    # Smoothing (Gaussian blur 15x15)
    blurred = smoothing(img)
    cv2.imwrite(str(out_dir / "lena_blurred_15x15.png"), blurred)

    # Rotation 90 CW
    rot90 = rotation(img, 90)
    cv2.imwrite(str(out_dir / "lena_rotated_90.png"), rot90)

    # Rotation 180
    rot180 = rotation(img, 180)
    cv2.imwrite(str(out_dir / "lena_rotated_180.png"), rot180)

    print(f"[Done] Saved outputs to: {out_dir}")
