import argparse, os, time, sys
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Tuple

# ---------- utils ----------
def ensure_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def resize_max(img, max_side=1000):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max(h, w) / max_side
    return cv2.resize(img, (int(w/s), int(h/s)))

def mem_bytes(arr): 
    return 0 if arr is None else int(arr.nbytes)

@dataclass
class Report:
    pipeline: str
    img1: str; img2: str
    kp1: int; kp2: int
    desc1_bytes: int; desc2_bytes: int
    detect_ms: float; match_ms: float; total_ms: float
    good_matches: int
    inliers: int
    inlier_ratio: float
    homography_found: bool

# ---------- pipelines ----------
def run_orb_bf(img1, img2, name1, name2, ratio=0.75, fast_threshold=20) -> Tuple[Report, np.ndarray]:
    t0 = time.time()
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=fast_threshold)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    t1 = time.time()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches_knn if m.distance < ratio * n.distance]

    H, inlier_count, inlier_ratio = None, 0, 0.0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_count = int(mask.sum())
            inlier_ratio = inlier_count / len(good)

    t2 = time.time()
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    rep = Report(
        pipeline="ORB+BF(Hamming)", img1=name1, img2=name2,
        kp1=len(kp1), kp2=len(kp2),
        desc1_bytes=mem_bytes(des1), desc2_bytes=mem_bytes(des2),
        detect_ms=(t1 - t0)*1000.0, match_ms=(t2 - t1)*1000.0, total_ms=(t2 - t0)*1000.0,
        good_matches=len(good), inliers=inlier_count, inlier_ratio=inlier_ratio,
        homography_found=H is not None
    )
    return rep, vis

def run_sift_flann(img1, img2, name1, name2, ratio=0.75) -> Tuple[Report, np.ndarray]:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("cv2.SIFT_create not found. Install opencv-contrib-python or conda-forge opencv.")
    t0 = time.time()
    sift = cv2.SIFT_create(nfeatures=4000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    t1 = time.time()

    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches_knn if m.distance < ratio * n.distance]

    H, inlier_count, inlier_ratio = None, 0, 0.0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_count = int(mask.sum())
            inlier_ratio = inlier_count / len(good)

    t2 = time.time()
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    rep = Report(
        pipeline="SIFT+FLANN", img1=name1, img2=name2,
        kp1=len(kp1), kp2=len(kp2),
        desc1_bytes=mem_bytes(des1), desc2_bytes=mem_bytes(des2),
        detect_ms=(t1 - t0)*1000.0, match_ms=(t2 - t1)*1000.0, total_ms=(t2 - t0)*1000.0,
        good_matches=len(good), inliers=inlier_count, inlier_ratio=inlier_ratio,
        homography_found=H is not None
    )
    return rep, vis

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Compare ORB+BF vs SIFT+FLANN matching.")
    ap.add_argument("--img1", type=str, help="First image path.")
    ap.add_argument("--img2", type=str, help="Second image path.")
    ap.add_argument("--folder", type=str, help="If set, take first two images in this folder.")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    ap.add_argument("--resize-max", type=int, default=1000, help="Resize so max side <= this (0 = no resize).")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test.")
    ap.add_argument("--fast-th", type=int, default=20, help="FAST threshold for ORB.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- choose input images ---
    if args.folder:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = sorted([
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.lower().endswith(exts)
        ])
        if len(files) < 2:
            print("Need at least two images in folder:", args.folder)
            sys.exit(1)
        img1_path, img2_path = files[0], files[1]
    else:
        if not (args.img1 and args.img2):
            print("Provide --img1 and --img2 or use --folder.")
            sys.exit(1)
        img1_path, img2_path = args.img1, args.img2

    # --- load and resize ---
    img1 = ensure_img(img1_path)
    img2 = ensure_img(img2_path)

    if args.resize_max and args.resize_max > 0:
        img1 = resize_max(img1, args.resize_max)
        img2 = resize_max(img2, args.resize_max)

    # --- run both pipelines ---
    rep_orb, vis_orb = run_orb_bf(
        img1, img2,
        os.path.basename(img1_path), os.path.basename(img2_path),
        ratio=args.ratio, fast_threshold=args.fast_th
    )
    rep_sift, vis_sift = run_sift_flann(
        img1, img2,
        os.path.basename(img1_path), os.path.basename(img2_path),
        ratio=args.ratio
    )

    # --- save results ---
    out_orb = os.path.join(args.outdir, "orb_matches.jpg")
    out_sift = os.path.join(args.outdir, "sift_matches.jpg")
    cv2.imwrite(out_orb, vis_orb)
    cv2.imwrite(out_sift, vis_sift)

    df = pd.DataFrame([asdict(rep_orb), asdict(rep_sift)])
    csv_path = os.path.join(args.outdir, "results.csv")
    df.to_csv(csv_path, index=False)

    # --- show summary ---
    def kb(x): return f"{x/1024:.1f} KB"
    print("\n=== RESULTS ===")
    for r in [rep_orb, rep_sift]:
        print(f"\nPipeline: {r.pipeline}")
        print(f"Images: {r.img1} vs {r.img2}")
        print(f"Keypoints: {r.kp1} / {r.kp2}")
        print(f"Descriptors mem: {kb(r.desc1_bytes)} / {kb(r.desc2_bytes)}")
        print(f"Times (ms): detect {r.detect_ms:.1f} | match {r.match_ms:.1f} | total {r.total_ms:.1f}")
        print(f"Good matches: {r.good_matches}")
        print(f"Inliers: {r.inliers}  (ratio {r.inlier_ratio:.3f})")
        print(f"Homography found: {r.homography_found}")
    print(f"\nSaved: {out_orb}, {out_sift}, {csv_path}")

if __name__ == "__main__":
    main()
