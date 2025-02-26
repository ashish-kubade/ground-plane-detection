import numpy as np
import cv2
import open3d as o3d
import sys

class PCD_Generator:
    def __init__(self):
        
        pass
        
    def run(self, rgb_image, depth_image, intrinsics=None) -> object:
        """Convert an RGB + depth image to a 3D point cloud."""
        if isinstance(rgb_image, str):
            rgb_image = cv2.imread(rgb_image)[:,:,::-1] #bgr to rgb
        if isinstance(depth_image, str):
            depth_image = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)

        height, width = depth_image.shape
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        if intrinsics == None:
            fx, fy = 500.0, 500.0  # Focal length in pixels
            cx, cy = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # Principal point
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

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
