from __future__ import annotations
import argparse
from pathlib import Path
import cv2 as cv
import numpy as np


# -------------------------------
# Hjelpefunksjoner
# -------------------------------
ASSETS_DIR = Path(__file__).parent / "assets"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Sørg for grått bilde."""
    if img is None:
        raise ValueError("Kunne ikke lese bildet. Sjekk sti/filnavn.")
    if len(img.shape) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def _save(img: np.ndarray, filename: str) -> Path:
    """Lagre bilde i output/-mappen."""
    outpath = OUTPUT_DIR / filename
    cv.imwrite(str(outpath), img)
    return outpath


# -------------------------------
# I. Sobel edge detection
# -------------------------------
def sobel_edge_detection(image: np.ndarray) -> Path:
    """
    Krav:
    - GaussianBlur(ksize=(3,3), sigmaX=0)
    - Sobel dx=1, dy=1, ksize=1
    - Lagre bildet
    """
    gray = _ensure_gray(image)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    # Sobel med dx=1, dy=1, ksize=1 (kombinerte retninger)
    sobel_xy64 = cv.Sobel(blurred, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)
    # Konverter til uint8 for visning/lagring
    sobel_xy = cv.convertScaleAbs(sobel_xy64)

    return _save(sobel_xy, "sobel_edge_detection.png")


# -------------------------------
# II. Canny edge detection
# -------------------------------
def canny_edge_detection(image: np.ndarray, threshold_1: int, threshold_2: int) -> Path:
    """
    Krav:
    - GaussianBlur(ksize=(3,3), sigmaX=0)
    - Canny(image, threshold1=50, threshold2=50) (fra oppgaven)
    - Lagre bildet
    """
    gray = _ensure_gray(image)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blurred, threshold_1, threshold_2)
    return _save(edges, f"canny_{threshold_1}_{threshold_2}.png")


# -------------------------------
# III. Template matching
# -------------------------------
def template_match(image: np.ndarray, template: np.ndarray, threshold: float = 0.9) -> Path:
    """
    Krav:
    - Konverter begge til grå
    - matchTemplate (TM_CCOEFF_NORMED), threshold=0.9
    - Marker treff med rød firkant (som i "Multiple Objects" eksempelet)
    - Lagre bildet
    Hot tip: både bilde og template må være grå.
    """
    img_gray = _ensure_gray(image)
    tmpl_gray = _ensure_gray(template)

    h, w = tmpl_gray.shape[:2]

    res = cv.matchTemplate(img_gray, tmpl_gray, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # Tegn på en BGR-kopi så rød blir synlig
    img_bgr = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

    # Tegn rektangler for alle posisjoner >= threshold
    for pt in zip(*loc[::-1]):  # (x, y)
        cv.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return _save(img_bgr, "template_match.png")


# -------------------------------
# IV. Resize med image pyramids
# -------------------------------
def resize(image: np.ndarray, scale_factor: int, up_or_down: str) -> Path:
    """
    Krav:
    - scale_factor: int (f.eks. 2)
    - up_or_down: "up" eller "down"
    - Bruk pyramid-funksjonene (pyrUp/pyrDown) scale_factor ganger
    - Lagre bildet
    """
    if scale_factor < 1:
        raise ValueError("scale_factor må være >= 1")

    img = image.copy()
    if up_or_down.lower() == "up":
        for _ in range(scale_factor):
            img = cv.pyrUp(img)
        suffix = f"upx{scale_factor}"
    elif up_or_down.lower() == "down":
        for _ in range(scale_factor):
            img = cv.pyrDown(img)
        suffix = f"downx{scale_factor}"
    else:
        raise ValueError('up_or_down må være "up" eller "down"')

    return _save(img, f"resize_{suffix}.png")


# -------------------------------
# CLI for rask testing
# -------------------------------
def _read_image(path_str: str) -> np.ndarray:
    return cv.imread(path_str, cv.IMREAD_UNCHANGED)


def main():
    parser = argparse.ArgumentParser(description="Assignment 3 – OpenCV verktøy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # sobel
    p_sobel = sub.add_parser("sobel")
    p_sobel.add_argument("--image", required=True, help="Sti til bilde")

    # canny
    p_canny = sub.add_parser("canny")
    p_canny.add_argument("--image", required=True)
    p_canny.add_argument("--t1", type=int, default=50)
    p_canny.add_argument("--t2", type=int, default=50)

    # template
    p_tmpl = sub.add_parser("template")
    p_tmpl.add_argument("--image", required=True)
    p_tmpl.add_argument("--template", required=True)
    p_tmpl.add_argument("--thr", type=float, default=0.9)

    # resize
    p_resize = sub.add_parser("resize")
    p_resize.add_argument("--image", required=True)
    p_resize.add_argument("--factor", type=int, default=2)
    p_resize.add_argument("--dir", choices=["up", "down"], required=True)

    args = parser.parse_args()

    if args.cmd == "sobel":
        img = _read_image(args.image)
        out = sobel_edge_detection(img)
    elif args.cmd == "canny":
        img = _read_image(args.image)
        out = canny_edge_detection(img, args.t1, args.t2)
    elif args.cmd == "template":
        img = _read_image(args.image)
        tmpl = _read_image(args.template)
        out = template_match(img, tmpl, args.thr)
    elif args.cmd == "resize":
        img = _read_image(args.image)
        out = resize(img, args.factor, args.dir)
    else:
        raise SystemExit("Ukjent kommando.")

    print(f"Lagret: {out}")


if __name__ == "__main__":
    main()
