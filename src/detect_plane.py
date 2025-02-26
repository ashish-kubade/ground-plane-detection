import cv2
import numpy as np
import open3d as o3d
from pcd_generation import PCD_Generator
import open3d as o3d

def detect_plane(image_path, depth_results, pcd_generator, use_defaults, save_pcd=False):


    rgb_image = cv2.imread(image_path)
    if depth_results["default_intrinsics"] or use_defaults:
        fx, fy = 525.0, 525.0    # Focal length in pixels
    else:
        fx, fy = depth_results["focal_length"], depth_results["focal_length"]

    cx, cy = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # Principal point
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    


    depth_image = depth_results['depth']
    pcd = pcd_generator.run(image_path, depth_image=depth_image, intrinsics=intrinsics, save_pcd=save_pcd)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000)

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
    return {
        'inlier_cloud': inlier_cloud,
        'outlier_cloud': outlier_cloud,
        'plane_model': plane_model,
        'pcd': pcd
    }
    