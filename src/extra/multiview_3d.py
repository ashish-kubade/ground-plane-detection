import cv2
import numpy as np
import glob

# Load all images from a folder
image_paths = sorted(glob.glob("/home/ashish/Documents/interviews/ground-plane-detection/data/ground_plane/3d_artist_space/*.png"))  # Ensure images are in order

images = [cv2.imread(path) for path in image_paths]
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# ORB feature detector
orb = cv2.ORB_create(1000)

# Store keypoints and descriptors for each image
keypoints_list, descriptors_list = [], []
for gray in gray_images:
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Function to match features between two images
def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Initialize the first camera pose (identity matrix)
poses = [np.hstack((np.eye(3), np.zeros((3, 1))))]

# Store 3D points
point_cloud = []

# Process image pairs
for i in range(len(images) - 1):
    print(f"Processing view {i} and {i + 1}...")

    # Match features
    matches = match_features(descriptors_list[i], descriptors_list[i + 1])
    pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches])

    # Estimate the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

    # Essential matrix from fundamental matrix
    K = np.eye(3)  # Assume an identity matrix for camera intrinsics
    K[0][0] = 10
    K[1][1] = 10
    K[0][2] = images[i].shape[1] / 2
    K[1][2] = images[i].shape[0] / 2

    E = K.T @ F @ K

    # Recover camera pose from the essential matrix
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

    # Projection matrices for triangulation
    P1 = poses[-1]
    P2 = np.hstack((R, t))
    poses.append(P2)

    # Triangulate points
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D[:3] / pts4D[3]

    # Append to the point cloud
    point_cloud.append(pts3D.T)

# Combine all 3D points
point_cloud = np.vstack(point_cloud)

# Save the 3D point cloud to a PLY file
with open("multi_view_output.ply", "w") as ply_file:
    ply_file.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(point_cloud.shape[0]))
    ply_file.write("property float x\nproperty float y\nproperty float z\nend_header\n")
    for point in point_cloud:
        ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")

print("3D point cloud saved as 'multi_view_output.ply'.")
