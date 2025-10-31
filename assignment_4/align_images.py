import cv2
import numpy as np
import os

def align_images(
    image_to_align_path: str,
    reference_image_path: str,
    max_features: int,
    good_match_percent: float,
    aligned_out_path: str,
    matches_out_path: str,
    method: str = "ORB",
):
    """
    Align image_to_align to reference_image using feature matching.

    Saves:
        aligned_out_path  -> warped aligned image
        matches_out_path  -> visualization of feature matches

    method: "ORB" or "SIFT"
    """

    # 1. Read images
    im = cv2.imread(image_to_align_path)
    im_ref = cv2.imread(reference_image_path)

    if im is None:
        raise FileNotFoundError(f"Could not load image at {image_to_align_path}")
    if im_ref is None:
        raise FileNotFoundError(f"Could not load image at {reference_image_path}")

    # Grayscale for feature detection
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_ref_gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)

    # 2. Detect keypoints + descriptors
    if method.upper() == "ORB":
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(im_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im_ref_gray, None)

        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        )
        raw_matches = matcher.match(descriptors1, descriptors2, None)

        # sort by distance (LOWER is better)
        matches = sorted(raw_matches, key=lambda x: x.distance, reverse=False)

        # keep best X%
        num_good = int(len(matches) * good_match_percent)
        if num_good < 4:
            num_good = min(4, len(matches))
        matches = matches[:num_good]

    else:
        sift = cv2.SIFT_create(nfeatures=max_features)
        keypoints1, descriptors1 = sift.detectAndCompute(im_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im_ref_gray, None)

        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < good_match_percent * n.distance:
                good_matches.append(m)

        matches = good_matches

    if len(matches) < 4:
        raise RuntimeError(
            f"Not enough matches to compute homography. Only {len(matches)} good matches."
        )

    # 3. Collect matched point coordinates
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts1[i, :] = keypoints1[match.queryIdx].pt  # from image_to_align
        pts2[i, :] = keypoints2[match.trainIdx].pt  # from reference

    # 4. Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # 5. Warp image_to_align so it lines up with reference
    height, width, _ = im_ref.shape
    im_aligned = cv2.warpPerspective(im, H, (width, height))

    # 6. Draw matches visualization
    matchesMask = mask.ravel().tolist() if mask is not None else None

    im_matches = cv2.drawMatches(
        im,
        keypoints1,
        im_ref,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 7. Make sure output/ exists
    os.makedirs(os.path.dirname(aligned_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(matches_out_path), exist_ok=True)

    # 8. Save results
    cv2.imwrite(aligned_out_path, im_aligned)
    cv2.imwrite(matches_out_path, im_matches)

    print(f"[OK] Aligned image saved to {aligned_out_path}")
    print(f"[OK] Matches visualization saved to {matches_out_path}")


# ---- RUN IMMEDIATELY WHEN SCRIPT IS EXECUTED ----
# No if __name__ == "__main__": guard here on purpose
image_to_align_path = "images/align_this.jpg"
reference_image_path = "images/reference_img.png"

# We'll start with ORB defaults
max_features = 1500
good_match_percent = 0.15
method = "ORB"

align_images(
    image_to_align_path=image_to_align_path,
    reference_image_path=reference_image_path,
    max_features=max_features,
    good_match_percent=good_match_percent,
    aligned_out_path="output/aligned.png",
    matches_out_path="output/matches.png",
    method=method,
)
