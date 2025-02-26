import cv2
import numpy as np
import sys
# Read the images
import sys
import os

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

image_root = sys.argv[1]
images = sorted(os.listdir(image_root))
image_paths = [os.path.join(image_root, image) for image in images if image.endswith('.png')]

images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

def find_homography(img1, img2):
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# Use the first image as the reference
reference_img = images[0]
height, width = reference_img.shape

# Store homographies relative to the first image
homographies = [np.eye(3)]

# Compute pairwise homographies
for i in range(1, len(images)):
    H = find_homography(reference_img, images[i])
    homographies.append(H)

# Warp each image to align with the reference
warped_images = [reference_img]
for i in range(1, len(images)):
    warped_img = cv2.warpPerspective(images[i], homographies[i], (width, height))
    warped_images.append(warped_img)

# Display results
for i, img in enumerate(warped_images):
    cv2.imshow(f"Warped Image {i+1}", img)

cv2.waitKey(0)
cv2.destroyAllWindows()