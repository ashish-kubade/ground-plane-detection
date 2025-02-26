import numpy as np
import cv2
import open3d as o3d

import sys
def depth_to_point_cloud(rgb_image, depth_image, intrinsics):
    """Convert an RGB + depth image to a 3D point cloud."""
    
    height, width = depth_image.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Normalize pixel coordinates
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0]  # (u - cx) / fx
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1]  # (v - cy) / fy
    
    # Get depth values (convert to meters if necessary)
    Z = depth_image.astype(np.float32)
    
    # Compute real-world coordinates (X, Y, Z)
    X = x * Z
    Y = y * Z
    
    # Stack into (N, 3) format
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    # Get RGB values
    colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0,1]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# Example usage
rgb_path = sys.argv[1]
depth_path = sys.argv[2]
rgb_image = cv2.imread(rgb_path)[:,:,::-1]  # Load RGB image
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Load depth map

# Define Camera Intrinsic Matrix (example values, replace with actual calibration)
fx, fy = 500.0, 500.0  # Focal length in pixels
cx, cy = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # Principal point
intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Generate point cloud
pcd = depth_to_point_cloud(rgb_image, depth_image, intrinsics)
o3d.io.write_point_cloud("output.ply", pcd)
o3d.visualization.draw_geometries([pcd])

