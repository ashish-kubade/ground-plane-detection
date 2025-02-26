import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os, sys
# Load 4 images
image_root = sys.argv[1]
image_names = os.listdir(image_root)
image_paths = [os.path.join(image_root, image) for image in image_names if image.endswith('.png')]
images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths]

# Initialize ORB detector
orb = cv2.ORB_create(5000)

# Extract features and descriptors
keypoints, descriptors = [], []
for img in images:
    kp, des = orb.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = [bf.match(descriptors[i], descriptors[i+1]) for i in range(3)]

# Function to extract matched points
def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

# Assume intrinsic camera matrix (K)
K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]])

# Estimate 3D structure
points_3d = []
for i in range(len(keypoints) - 1):
    pts1, pts2 = get_matched_points(keypoints[i], keypoints[i+1], matches[i])
    E, _ = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Triangulation
    proj1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    proj2 = np.dot(K, np.hstack((R, t)))
    points4D = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    points4D /= points4D[3]
    points_3d.append(points4D[:3].T)

# Combine all 3D points
points_3d = np.vstack(points_3d)

# Plot the 3D point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
