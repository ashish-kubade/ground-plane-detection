import cv2
import numpy as np
import os
import sys


def detect_ground_plane_uncalibrated(image_paths):
    """
    Detects the ground plane from multiple UNCALIBRATED images using homography 
    and RANSAC.  This approach assumes a planar scene and uses homographies 
    directly without 3D reconstruction.

    Args:
        image_paths: A list of paths to the images.

    Returns:
        A tuple containing:
            - The homography from the reference image to the "rectified" ground plane view.
            - A list of inlier points used to estimate the homography.
        Returns None if ground plane detection fails.
    """

    images = [cv2.imread(path) for path in image_paths]
    if not all(img is not None for img in images):
        print("Error: Could not read all images.")
        return None

    # Feature detection and matching (using ORB for speed)
    orb = cv2.ORB_create()
    keypoints = []
    descriptors = []
    for img in images:
        k, d = orb.detectAndCompute(img, None)
        keypoints.append(k)
        descriptors.append(d)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Choose a reference image (e.g., the first one)
    ref_idx = 0
    ref_keypoints = keypoints[ref_idx]
    ref_descriptors = descriptors[ref_idx]

    # We will accumulate matched points from all images to estimate a single homography
    all_matched_keypoints1 = []
    all_matched_keypoints2 = []

    for i in range(len(images)):
        if i == ref_idx:
            continue

        matches = matcher.match(ref_descriptors, descriptors[i])
        matched_keypoints1 = [ref_keypoints[m.queryIdx].pt for m in matches]
        matched_keypoints2 = [keypoints[i][m.trainIdx].pt for m in matches]

        MIN_MATCHES = 10  # Minimum matches for homography calculation
        if len(matches) < MIN_MATCHES:
            print(f"Not enough matches for image {i+1}.")
            continue # Skip this image if not enough matches

        all_matched_keypoints1.extend(matched_keypoints1)  # Accumulate points
        all_matched_keypoints2.extend(matched_keypoints2)

    if not all_matched_keypoints1:  # Check if any matches were found at all
        print("No matches found across all images.")
        return None

    # Convert to numpy arrays for findHomography
    src_pts = np.float32(all_matched_keypoints1).reshape(-1, 1, 2)
    dst_pts = np.float32(all_matched_keypoints2).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography estimation failed.")
        return None

    inlier_points = src_pts[mask.ravel() == 1]  # Get inlier points

    return H, inlier_points


# Example usage (replace with your image paths)
image_root = sys.argv[1]
images = sorted(os.listdir(image_root))
image_paths = [os.path.join(image_root, image) for image in images if image.endswith('.png')] # Replace with your image paths

result = detect_ground_plane_uncalibrated(image_paths)

if result:
    H, inlier_points = result
    print("Homography to Ground Plane View:\n", H)
    print("Number of Inlier Points:", len(inlier_points))

    # Example: Warp the reference image to the "rectified" ground plane view
    ref_image = cv2.imread(image_paths[0])
    warped_image = cv2.warpPerspective(ref_image, H, (ref_image.shape[1], ref_image.shape[0]))
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)


else:
    print("Ground plane detection failed.")