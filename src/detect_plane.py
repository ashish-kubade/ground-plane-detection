import cv2
import numpy as np
import open3d as o3d
import sys
import os

image_root = sys.argv[1]
images = sorted(os.listdir(image_root))
image_paths = [os.path.join(image_root, image) for image in images]

# Step 2: Feature detection and matching
sift = cv2.SIFT_create()
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

keypoints_list, descriptors_list = [], []
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Match features between the first two images with Lowe's ratio test
good_matches = []
matches = flann.knnMatch(descriptors_list[0], descriptors_list[1], k=2)
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
src_pts = np.float32([keypoints_list[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_list[1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Step 3: Estimate Essential matrix and recover pose
E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)
_, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)

# Triangulate points to get 3D structure
proj1 = np.eye(3, 4)
proj2 = np.hstack((R, t))
points_4d_hom = cv2.triangulatePoints(proj1, proj2, src_pts, dst_pts)
points_3d = points_4d_hom[:3] / points_4d_hom[3]

# Convert points to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T)

# Denoise point cloud
pcd = pcd.voxel_down_sample(voxel_size=0.01)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)

# Step 4: Fit a plane using RANSAC with higher iterations
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=3000)

# Output plane equation
A, B, C, D = plane_model
print(f"Plane equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0")

# Visualize the plane and point cloud
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
