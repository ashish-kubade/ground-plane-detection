import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys

# Load 4 RGB and depth images
image_root = sys.argv[1]
image_names = os.listdir(image_root)
image_paths = [os.path.join(image_root, image) for image in image_names if image.endswith('.png')]
rgb_images = [cv2.imread(img) for img in image_paths]
depth_paths = [img.replace('.png', '_depth.jpg') for img in image_paths]
depth_images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in depth_paths]

# Intrinsic camera matrix (K)
K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

# Convert depth to 3D points
def depth_to_point_cloud(rgb, depth, K):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth[v, u] / 1000.0  # Assuming depth in millimeters
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(rgb[v, u])

    return np.array(points), np.array(colors)

# Collect 3D points and colors
all_points = []
all_colors = []

for i in range(4):
    points, colors = depth_to_point_cloud(rgb_images[i], depth_images[i], K)
    all_points.append(points)
    all_colors.append(colors)

all_points = np.vstack(all_points)
all_colors = np.vstack(all_colors)

# Plot the 3D point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c=all_colors / 255.0, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
