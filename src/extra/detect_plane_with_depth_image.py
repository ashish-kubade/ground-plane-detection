import cv2
import numpy as np
import open3d as o3d
import os
import sys

# Function to estimate ground plane from depth map and RGB image
def estimate_ground_plane(rgb_image, depth_map):
    # Convert depth map to point cloud
    height, width = depth_map.shape
    fx, fy = 1.0, 1.0 # Example focal lengths (adjust based on your camera)
    cx, cy = width // 2, height // 2

    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z =  1 / (depth_map[v, u] + 1e-5)  # Convert depth to meters
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(rgb_image[v, u] / 255.0)

    points = np.array(points)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    # Estimate ground plane using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=3000)
    A, B, C, D = plane_model

    print(f"Estimated ground plane equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0")

    # Visualize
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return plane_model


image_path = sys.argv[1]
depth_map_path = sys.argv[2]
# Example usage
rgb_image = cv2.imread(image_path)
depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

plane_model = estimate_ground_plane(rgb_image, depth_map)
